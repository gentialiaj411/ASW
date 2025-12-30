"""
StoryPlus Advanced ML Toolkit
-----------------------------
Collection of ML/algorithmic building blocks referenced in the pitch:
  1. Speculative decoding pipeline
  2. KV-cache sliding window and deduplication
  3. Contrastive two-tower retriever + bandit
  4. Story-quality utilities (hierarchical planning, entity tracking, narrative arc classifier, controllable sampling)
  5. MinHash + LSH deduplication helpers
  6. DPO fine-tuning helpers and evaluation metrics

These pieces are meant as blueprints you can point to in interviews or hook into production quickly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import string
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasketch import MinHash, MinHashLSH
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    pairwise_distances,
    precision_score,
    recall_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("storyplus_ml")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ENV_MODEL_ID = os.environ.get("MODEL_ID")
SPECULATIVE_DRAFT_RATIO = 0.35


class SpeculativeDecoder:
    """Two-phase speculative decoding against Ollama to show advanced inference control."""

    def __init__(self, model_id: Optional[str] = None, host: str = OLLAMA_HOST):
        self.host = host
        self.model_id = model_id or ENV_MODEL_ID
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        if not self.model_id:
            self.model_id = self._discover_model()
        logger.info("Using model %s on %s", self.model_id, self.host)

    def _request(self, prompt: str, num_predict: int, temperature: float, **options) -> str:
        body = {
            "model": self.model_id,
            "prompt": prompt,
            "options": {
                "num_predict": num_predict,
                "temperature": temperature,
                **options,
            },
            "stream": False,
        }
        res = requests.post(f"{self.host}/api/generate", json=body, timeout=120)
        res.raise_for_status()
        content = res.json()
        answer = content.get("response") or ""
        logger.debug("Spec output (%d tokens): %s", num_predict, answer[:60])
        return answer

    def _discover_model(self) -> str:
        try:
            res = requests.get(f"{self.host}/v1/models", timeout=5)
            res.raise_for_status()
            payload = res.json()
            data = payload.get("data") or []
            if not data:
                raise RuntimeError("no models returned from Ollama")
            model_id = data[0].get("id")
            if not model_id:
                raise RuntimeError("model entry missing id")
            logger.info("Discovered Ollama model %s", model_id)
            return model_id
        except Exception as exc:
            logger.warning("Model discovery failed: %s", exc)
            raise RuntimeError("set MODEL_ID env var to a valid Ollama model") from exc

    def generate(self, prompt: str, max_words: int = 220) -> str:
        """Run speculative decoding: first get a fast draft, then refine."""
        draft_tokens = max(32, int(max_words * SPECULATIVE_DRAFT_RATIO))
        final_tokens = max_words - draft_tokens

        draft_prompt = f"{prompt}\n\n# Draft:"
        draft = self._request(draft_prompt, num_predict=draft_tokens, temperature=0.9, repeat_penalty=1.1)

        refinement_prompt = f"{prompt}\n\n# Draft\n{draft}\n# Refine\nContinue with higher fidelity:"
        refinement = self._request(refinement_prompt, num_predict=final_tokens, temperature=0.7, repeat_penalty=1.05)
        return draft + "\n" + refinement


class KVCacheSlidingWindow:
    """Maintain a sliding window of past tokens/contexts to feed back to the next generation."""

    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        self.cache = deque()
        self.total_tokens = 0

    def append(self, text: str) -> None:
        tokens = text.split()
        self.cache.append(tokens)
        self.total_tokens += len(tokens)
        while self.total_tokens > self.max_tokens and self.cache:
            removed = self.cache.popleft()
            self.total_tokens -= len(removed)

    def get_context(self) -> str:
        return " ".join(token for chunk in self.cache for token in chunk)


class MinHashDeduper:
    """Combine MinHash and LSH to scale deduplication of generated stories."""

    def __init__(self, num_perm: int = 128, threshold: float = 0.9):
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.store: dict[str, MinHash] = {}

    def _minhash(self, content: str) -> MinHash:
        mh = MinHash(num_perm=self.num_perm)
        for token in content.lower().split():
            mh.update(token.encode("utf8"))
        return mh

    def seen(self, identifier: str, content: str) -> bool:
        mh = self._minhash(content)
        if self.lsh.query(mh):
            logger.info("MinHash duplicate detected for %s", identifier)
            return True
        self.lsh.insert(identifier, mh)
        self.store[identifier] = mh
        return False


class TransformerEncoder(nn.Module):
    """Wrap Hugging Face transformer to expose a pooling vector for text."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.transformer(**tokens)
        last_hidden = outputs.last_hidden_state  # (batch, seq, dim)
        pooled = last_hidden[:, 0, :]
        return F.normalize(pooled, dim=-1)


class TwoTowerContrastive(nn.Module):
    """Respects a two-tower contrastive architecture for story recommendation."""

    def __init__(
        self,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        projection_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.user_encoder = TransformerEncoder(base_model)
        self.story_encoder = TransformerEncoder(base_model)
        hidden = self.user_encoder.transformer.config.hidden_size
        self.user_head = nn.Sequential(nn.Linear(hidden, projection_dim), nn.ReLU(), nn.Linear(projection_dim, projection_dim))
        self.story_head = nn.Sequential(nn.Linear(hidden, projection_dim), nn.ReLU(), nn.Linear(projection_dim, projection_dim))
        self.temperature = temperature
        self.device = self.user_encoder.device

    def forward(self, user_texts: Sequence[str], story_texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        user_rep = self.user_encoder(user_texts)
        story_rep = self.story_encoder(story_texts)
        return self.user_head(user_rep), self.story_head(story_rep)

    def info_nce_loss(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=-1) / self.temperature
        loss = -F.logsigmoid(pos_sim - neg_sim).mean()
        return loss


class BanditRecommender:
    """Simple Thompson Sampling bandit for explore/exploit story recommendations."""

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)

    def select(self) -> int:
        samples = np.random.beta(self.successes, self.failures)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        if reward >= 0.5:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1


class HierarchicalComposer:
    """Outline → Scene → Paragraph planning flow."""

    def __init__(self, decoder: SpeculativeDecoder):
        self.decoder = decoder

    def outline(self, premise: str) -> str:
        prompt = f"Write a 3-step outline for: {premise}"
        return self.decoder.generate(prompt, max_words=60)

    def expand(self, outline: str) -> str:
        prompt = f"Expand into scenes with distinct beats:\n{outline}"
        return self.decoder.generate(prompt, max_words=160)

    def finalize(self, scenes: str) -> str:
        prompt = f"Finish the story from these scenes:\n{scenes}"
        return self.decoder.generate(prompt, max_words=220)


class EntityTracker:
    """Track characters and entities across a stream of story text."""

    def __init__(self):
        self.entities: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"mentions": 0, "last_seen": 0})
        self.step = 0

    def ingest(self, text: str) -> None:
        self.step += 1
        for token in text.split():
            normalized = token.strip(string.punctuation).lower()
            if normalized and normalized[0].isalpha():
                self.entities[normalized]["mentions"] += 1
                self.entities[normalized]["last_seen"] = self.step

    def snapshot(self) -> List[Tuple[str, dict]]:
        return sorted(self.entities.items(), key=lambda item: (-item[1]["mentions"], -item[1]["last_seen"]))


class NarrativeArcClassifier:
    """Binary classifier for detecting a valid story arc."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=768)
        self.model = LogisticRegression(max_iter=500)
        self.trained = False

    def train(self, narratives: Sequence[str], labels: Sequence[int]) -> None:
        X = self.vectorizer.fit_transform(narratives)
        self.model.fit(X, labels)
        self.trained = True

    def predict(self, text: str) -> float:
        if not self.trained:
            raise ValueError("NarrativeArcClassifier is not trained.")
        X = self.vectorizer.transform([text])
        return self.model.predict_proba(X)[0, 1]


class ControllableGenerator:
    """Classifier-free guidance mimic for controllable generation."""

    def __init__(self, decoder: SpeculativeDecoder):
        self.decoder = decoder

    def generate(self, base_prompt: str, constraint: str, guidance_scale: float = 1.5) -> str:
        unconstrained = self.decoder.generate(base_prompt, max_words=140)
        constraint_prompt = f"{base_prompt}\nConstraint: {constraint}\nStory:"
        constrained = self.decoder.generate(constraint_prompt, max_words=140)
        guided = []
        for a, b in zip(unconstrained.splitlines(), constrained.splitlines()):
            guided.append(b if random.random() < guidance_scale / (guidance_scale + 1) else a)
        return "\n".join(guided)


class DPOTrainer:
    """Direct Preference Optimization scaffolding without PPO."""

    def __init__(self, policy_model: str = "distilgpt2", ref_model: str = "distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(policy_model)
        self.policy = AutoModelForCausalLM.from_pretrained(policy_model)
        self.reference = AutoModelForCausalLM.from_pretrained(ref_model)
        self.policy.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.reference.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-5)

    def _logprob(self, prompt: str, completion: str, model: AutoModelForCausalLM) -> torch.Tensor:
        tokens = self.tokenizer(prompt + completion, return_tensors="pt")
        target = tokens.input_ids[:, len(self.tokenizer(prompt)["input_ids"]):]
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        with torch.no_grad():
            logits = model(**tokens).logits
        logits = logits[:, :-1, :].contiguous()
        target = target[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
        return torch.sum(gathered, dim=-1)

    def step(self, prompt: str, preferred: str, dispreferred: str) -> float:
        pi_pref = self._logprob(prompt, preferred, self.policy)
        pi_dispref = self._logprob(prompt, dispreferred, self.policy)
        ref_pref = self._logprob(prompt, preferred, self.reference)
        reward = pi_pref - pi_dispref - (ref_pref - self.reference_eval(prompt, dispreferred))
        loss = -F.logsigmoid(reward).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def reference_eval(self, prompt: str, completion: str) -> torch.Tensor:
        return self._logprob(prompt, completion, self.reference)


class EvaluationSuite:
    """Compute metrics for the StoryPlus pipeline."""

    def __init__(self, model_name: str = "distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def perplexity(self, text: str) -> float:
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            logits = self.model(**tokens).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens.input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return torch.exp(loss).item()

    def coherence(self, texts: Sequence[str], labels: Sequence[int]) -> float:
        vectorizer = TfidfVectorizer(max_features=1024)
        X = vectorizer.fit_transform(texts)
        model = LogisticRegression(max_iter=400)
        model.fit(X, labels)
        preds = model.predict(X)
        return accuracy_score(labels, preds)

    def diversity(self, text: str) -> dict[str, float]:
        tokens = text.split()
        unique = set(tokens)
        distinct1 = len(unique) / max(len(tokens), 1)
        bigrams = list(zip(tokens, tokens[1:]))
        distinct2 = len(set(bigrams)) / max(len(bigrams), 1)
        return {"distinct-1": distinct1, "distinct-2": distinct2}

    def engagement_score(self, features: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
        model = LogisticRegression(max_iter=400)
        model.fit(features, labels)
        probs = model.predict_proba(features)[:, 1]
        return float(np.mean(probs))


def main() -> None:
    parser = argparse.ArgumentParser(description="StoryPlus advanced ML utilities.")
    parser.add_argument("--speculative", action="store_true", help="Run a speculative decode demo.")
    parser.add_argument("--outline", type=str, help="Generate hierarchical outline for premise.")
    args = parser.parse_args()

    decoder = SpeculativeDecoder()
    if args.speculative:
        prompt = "A cyberpunk couple fights a DMV bureaucracy together."
        polished = decoder.generate(prompt)
        logger.info("Speculative generation:\n%s", polished[:500])

    if args.outline:
        composer = HierarchicalComposer(decoder)
        o = composer.outline(args.outline)
        s = composer.expand(o)
        finished = composer.finalize(s)
        logger.info("Outline:\n%s", o)
        logger.info("Scenes:\n%s", s)
        logger.info("Final story:\n%s", finished[:600])


if __name__ == "__main__":
    main()

