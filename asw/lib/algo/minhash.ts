const PRIME = 18446744073709551533n;
const MAX_HASH = PRIME;
const DEFAULT_NUM_PERM = 64;
const DEFAULT_BAND_SIZE = 8;
const DEFAULT_THRESHOLD = 0.88;

function hashToken(token: string): bigint {
  let hash = 1469598103934665603n;
  for (const ch of token) {
    hash ^= BigInt(ch.codePointAt(0) ?? 0);
    hash *= 1099511628211n;
    hash &= 0xFFFFFFFFFFFFFFFFn;
  }
  return hash;
}

const HASH_COEFFS = Array.from({ length: DEFAULT_NUM_PERM }, (_, idx) => ({
  a: BigInt(6364136223846793005n * BigInt(idx + 1)) % PRIME || 1n,
  b: BigInt(1442695040888963407n * BigInt(idx + 1)) % PRIME || 1n,
}));

const tokenize = (text: string): Set<string> =>
  new Set(
    text
      .toLowerCase()
      .split(/\W+/)
      .filter(Boolean)
      .slice(0, 256)
  );

const jaccard = (a: Set<string>, b: Set<string>) => {
  if (!a.size || !b.size) return 0;
  let intersection = 0;
  for (const token of a) {
    if (b.has(token)) intersection++;
  }
  if (!intersection) return 0;
  const union = new Set([...a, ...b]).size;
  return union === 0 ? 0 : intersection / union;
};

type Signature = bigint[];

export interface MinHashLSHDeduperConfig {
  threshold?: number;
  bandSize?: number;
  numPermutations?: number;
  maxEntries?: number;
}

export class MinHashLSHDeduper {
  private buckets = new Map<string, Set<string>>();
  private idToBands = new Map<string, string[]>();
  private tokenStore = new Map<string, Set<string>>();
  private maxEntries: number;
  private threshold: number;
  private bandSize: number;
  private numPermutations: number;
  private bandCount: number;

  constructor(config: MinHashLSHDeduperConfig = {}) {
    this.threshold = config.threshold ?? DEFAULT_THRESHOLD;
    this.bandSize = config.bandSize ?? DEFAULT_BAND_SIZE;
    this.numPermutations = config.numPermutations ?? DEFAULT_NUM_PERM;
    if (this.numPermutations % this.bandSize !== 0) {
      throw new Error("numPermutations must be divisible by bandSize");
    }
    this.bandCount = this.numPermutations / this.bandSize;
    this.maxEntries = config.maxEntries ?? 512;
  }

  private computeSignature(text: string): { signature: Signature; tokens: Set<string> } {
    const tokens = tokenize(text);
    const signature: bigint[] = new Array(this.numPermutations).fill(MAX_HASH);
    if (!tokens.size) {
      return { signature: signature.map(() => 0n), tokens };
    }
    for (const token of tokens) {
      const hashed = hashToken(token);
      for (let i = 0; i < this.numPermutations; i++) {
        const { a, b } = HASH_COEFFS[i];
        const value = (a * hashed + b) % PRIME;
        if (value < signature[i]) {
          signature[i] = value;
        }
      }
    }
    return { signature, tokens };
  }

  private bandKeys(signature: Signature): string[] {
    const bands: string[] = [];
    for (let band = 0; band < this.bandCount; band++) {
      const start = band * this.bandSize;
      const slice = signature
        .slice(start, start + this.bandSize)
        .map((val) => val.toString(16).padStart(16, "0"))
        .join(".");
      bands.push(`${band}:${slice}`);
    }
    return bands;
  }

  private prune() {
    while (this.tokenStore.size > this.maxEntries) {
      const [oldest] = this.tokenStore.keys();
      if (!oldest) break;
      const bandKeys = this.idToBands.get(oldest) ?? [];
      for (const key of bandKeys) {
        const bucket = this.buckets.get(key);
        bucket?.delete(oldest);
        if (bucket && bucket.size === 0) {
          this.buckets.delete(key);
        }
      }
      this.tokenStore.delete(oldest);
      this.idToBands.delete(oldest);
    }
  }

  public isDuplicate(text: string): boolean {
    const { signature, tokens } = this.computeSignature(text);
    const bands = this.bandKeys(signature);
    for (const bandKey of bands) {
      const bucket = this.buckets.get(bandKey);
      if (!bucket) continue;
      for (const candidate of bucket) {
        const storedTokens = this.tokenStore.get(candidate);
        if (!storedTokens) continue;
        if (jaccard(tokens, storedTokens) >= this.threshold) {
          return true;
        }
      }
    }
    return false;
  }

  public remember(id: string, text: string) {
    const { signature, tokens } = this.computeSignature(text);
    const bands = this.bandKeys(signature);
    for (const key of bands) {
      const bucket = this.buckets.get(key) ?? new Set();
      bucket.add(id);
      this.buckets.set(key, bucket);
    }
    this.tokenStore.set(id, tokens);
    this.idToBands.set(id, bands);
    this.prune();
  }
}

