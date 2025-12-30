## StoryPlus
HACK CMU PROJECT: AI Storytelling Platform

Overview: 
An AI-powered storytelling app that generates infinitely long stories in real-time, creating a TikTok-style feed of narratives.

-Key Features: 
- AI Story Generation:
    - Fine-tuned LLaMa 3.1 by training it on 100k+ story dataset
    - Real-time streaming generation with Server-Sent Events(SSE)
    - Stories can go on indefinitely by passing summaries between generations while maintaining consistency and focus on each story
    - Deduplication using SimHash algorithm to prevent repetetive content
- Text-to-Speech Integration:
    - OpenAI TTS API with 8 different voices and synchronized playback 
- Recommendation System:
    - Sentence-transformers embeddings and cosine similarity matching for content recommendations
    - Automatic keyword extraction from story content

Advanced ML Toolkit:

- **Speculative decoding** with a two-phase draft + refinement setup that you can run via `scripts/storyplus_ml.py`.
- **KV-cache sliding window** plus **MinHash + LSH deduplication** classes for preventing repetition at scale.
- **Two-Tower contrastive retriever** training skeleton, Thompson sampling bandit, and narrative arc/coherence evaluation metrics for recruiter-focused talking points.
- **Direct Preference Optimization (DPO)** and evaluation hooks to demonstrate modern RLHF knowledge.

Run `python scripts/storyplus_ml.py --help` to explore the pipelines and demos.

