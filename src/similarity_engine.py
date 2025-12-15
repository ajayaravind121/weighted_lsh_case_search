import pandas as pd
import numpy as np
import random
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


class WeightedLSH:
    def __init__(self, csv_path="data/mtsamples.csv",
                 max_docs=1000, num_hashes=100, b=20, r=5):
        self.csv_path = csv_path
        self.max_docs = max_docs
        self.num_hashes = num_hashes
        self.b = b
        self.r = r
        self.prime = 4294967311

        self.df = None
        self.vectorizer = None
        self.X = None
        self.signatures = None
        self.buckets = None
        self.hashfuncs = None

    # STEP 1: Load data
    def load_data(self):
        df = pd.read_csv(self.csv_path).head(self.max_docs)

        # Make sure all expected columns exist
        for col in ["description", "medical_specialty",
                    "sample_name", "transcription", "keywords"]:
            if col not in df.columns:
                df[col] = ""

        # Text used for similarity (combine description + transcription + keywords)
        df["text"] = (
            df["description"].fillna("").astype(str)
            + " "
            + df["transcription"].fillna("").astype(str)
            + " "
            + df["keywords"].fillna("").astype(str)
        )

        self.df = df.reset_index(drop=True)

    # STEP 2: TF-IDF
    def build_tfidf(self):
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.df["text"])

    # STEP 3: Build weighted hash functions
    def build_hashes(self):
        rng = random.Random(42)
        self.hashfuncs = [
            (rng.randint(1, self.prime - 1), rng.randint(0, self.prime - 1))
            for _ in range(self.num_hashes)
        ]

    # STEP 4: Weighted MinHash signature for a single vector
    def weighted_minhash(self, vec):
        nz = vec.nonzero()[1]
        weights = vec.data
        sig = []

        for a, b in self.hashfuncs:
            best = None
            best_hash = 0

            if len(nz) == 0:
                sig.append(0)
                continue

            for i, w in zip(nz, weights):
                h = (a * int(i) + b) % self.prime
                key = h / (w + 1e-9)
                if best is None or key < best:
                    best = key
                    best_hash = h

            sig.append(best_hash)

        return sig

    # STEP 5: Build signatures for all docs
    def build_signatures(self):
        sigs = []
        for i in range(self.X.shape[0]):
            sigs.append(self.weighted_minhash(self.X[i]))
        self.signatures = np.array(sigs, dtype=np.uint64)

    # STEP 6: LSH banding
    def build_lsh(self):
        buckets = defaultdict(list)
        for idx, sig in enumerate(self.signatures):
            for bi in range(self.b):
                start = bi * self.r
                band = tuple(sig[start:start + self.r])
                buckets[(bi, hash(band))].append(idx)
        self.buckets = buckets

    # Build full index
    def build(self):
        self.load_data()
        self.build_tfidf()
        self.build_hashes()
        self.build_signatures()
        self.build_lsh()

    # STEP 7: Query (with brute-force fallback)
    def query(self, text, topk=5):
        if self.vectorizer is None:
            raise RuntimeError("Call build() first.")

        qvec = self.vectorizer.transform([text])
        qsig = self.weighted_minhash(qvec)

        candidates = set()
        for bi in range(self.b):
            start = bi * self.r
            band = tuple(qsig[start:start + self.r])
            key = (bi, hash(band))
            if key in self.buckets:
                candidates.update(self.buckets[key])

        # If LSH yields no candidates, fall back to ALL docs
        if not candidates:
            candidates = set(range(self.X.shape[0]))

        q = qvec.toarray()
        results = []

        for c in candidates:
            v = self.X[c].toarray()
            num = np.minimum(q, v).sum()
            den = np.maximum(q, v).sum()
            sim = float(num / den) if den != 0 else 0.0

            row = self.df.iloc[c]
            results.append({
                "similarity": sim,
                "description": row["description"],
                "medical_specialty": row["medical_specialty"],
                "sample_name": row["sample_name"],
                "transcription": row["transcription"],
                "keywords": row["keywords"],
            })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:topk]
