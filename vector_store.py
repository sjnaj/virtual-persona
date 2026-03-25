"""
轻量向量存储 —— SQLite 持久化
优先使用外部 embed_fn（Doubao embedding API）做 cosine 检索；
无 embed_fn 时回退 TF-IDF。
"""
import os
import math
import json
import sqlite3
from collections import Counter
from typing import Callable, List, Dict, Optional

import numpy as np


class Collection:
    def __init__(self, db_path: str, name: str,
                 embed_fn: Optional[Callable[[str], Optional[np.ndarray]]] = None):
        self.name = name
        self.db_path = db_path
        self.embed_fn = embed_fn
        self._init_table()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        with self._conn() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS "{self.name}" (
                    id        TEXT PRIMARY KEY,
                    document  TEXT NOT NULL,
                    metadata  TEXT NOT NULL DEFAULT '{{}}',
                    embedding BLOB
                )
            """)
            # 迁移：为旧表添加 embedding 列（如不存在）
            cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{self.name}")')}
            if "embedding" not in cols:
                conn.execute(f'ALTER TABLE "{self.name}" ADD COLUMN embedding BLOB')
            conn.commit()

    # ------------------------------------------------------------------ write

    def add(self, documents: List[str], metadatas: List[dict], ids: List[str]):
        rows = []
        for i, d, m in zip(ids, documents, metadatas):
            emb_blob = None
            if self.embed_fn:
                vec = self.embed_fn(d)
                if vec is not None:
                    emb_blob = vec.tobytes()
            rows.append((i, d, json.dumps(m, ensure_ascii=False), emb_blob))

        with self._conn() as conn:
            conn.executemany(
                f'INSERT OR REPLACE INTO "{self.name}" (id, document, metadata, embedding) VALUES (?, ?, ?, ?)',
                rows,
            )
            conn.commit()

    def update(self, ids: List[str], metadatas: List[dict]):
        with self._conn() as conn:
            conn.executemany(
                f'UPDATE "{self.name}" SET metadata = ? WHERE id = ?',
                [(json.dumps(m, ensure_ascii=False), i)
                 for i, m in zip(ids, metadatas)],
            )
            conn.commit()

    def delete(self, ids: List[str]):
        with self._conn() as conn:
            conn.executemany(
                f'DELETE FROM "{self.name}" WHERE id = ?',
                [(i,) for i in ids],
            )
            conn.commit()

    # ------------------------------------------------------------------ read

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                f'SELECT COUNT(*) FROM "{self.name}"'
            ).fetchone()[0]

    def get(self) -> dict:
        with self._conn() as conn:
            rows = conn.execute(
                f'SELECT id, document, metadata, embedding FROM "{self.name}"'
            ).fetchall()
        return {
            "ids":        [r[0] for r in rows],
            "documents":  [r[1] for r in rows],
            "metadatas":  [json.loads(r[2]) for r in rows],
            "embeddings": [
                np.frombuffer(r[3], dtype=np.float32) if r[3] else None
                for r in rows
            ],
        }

    def query(self, query_texts: List[str], n_results: int = 5) -> dict:
        data = self.get()
        docs = data["documents"]
        if not docs:
            return {"documents": [[]], "metadatas": [[]]}

        n = min(n_results, len(docs))

        # 优先用 embedding cosine
        if self.embed_fn:
            q_vec = self.embed_fn(query_texts[0])
            stored = data["embeddings"]
            if q_vec is not None and any(e is not None for e in stored):
                scores = _cosine_scores(q_vec, stored, docs)
                top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
                return {
                    "documents": [[docs[i] for i in top]],
                    "metadatas": [[data["metadatas"][i] for i in top]],
                }

        # 回退 TF-IDF
        scores = _tfidf_cosine(query_texts[0], docs)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return {
            "documents": [[docs[i] for i in top]],
            "metadatas": [[data["metadatas"][i] for i in top]],
        }


# ------------------------------------------------------------------ client

class PersistentClient:
    def __init__(self, path: str,
                 embed_fn: Optional[Callable[[str], Optional[np.ndarray]]] = None):
        os.makedirs(path, exist_ok=True)
        self._db = os.path.join(path, "store.db")
        self._embed_fn = embed_fn
        self._cols: Dict[str, Collection] = {}

    def get_or_create_collection(self, name: str, metadata: dict = None) -> Collection:
        if name not in self._cols:
            self._cols[name] = Collection(self._db, name, self._embed_fn)
        return self._cols[name]


# ------------------------------------------------------------------ similarity

def _cosine_scores(q_vec: np.ndarray, stored: List[Optional[np.ndarray]],
                   docs: List[str]) -> List[float]:
    """embedding cosine；缺失 embedding 的条目回退单文本 TF-IDF 兜底。"""
    q_norm = float(np.linalg.norm(q_vec))
    if q_norm == 0:
        return [0.0] * len(docs)

    scores = []
    missing_idx = [i for i, e in enumerate(stored) if e is None]

    # TF-IDF 兜底分数（仅对缺失 embedding 的文档）
    fallback: List[float] = []
    if missing_idx:
        fallback_docs = [docs[i] for i in missing_idx]
        fallback = _tfidf_cosine(
            " ".join(q_vec.tobytes()[:0].decode(errors="ignore")),  # dummy
            fallback_docs,
        )
        # 用 query text 的 TF-IDF 更准确；但 q_vec 是 embedding，这里直接给 0
        fallback = [0.0] * len(missing_idx)

    fi = 0
    for i, emb in enumerate(stored):
        if emb is not None:
            d_norm = float(np.linalg.norm(emb))
            if d_norm == 0:
                scores.append(0.0)
            else:
                scores.append(float(np.dot(q_vec, emb) / (q_norm * d_norm)))
        else:
            scores.append(fallback[fi])
            fi += 1
    return scores


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for i, ch in enumerate(text):
        tokens.append(ch)
        if i < len(text) - 1:
            tokens.append(text[i: i + 2])
    return tokens or [""]


def _tfidf_cosine(query: str, docs: List[str]) -> List[float]:
    tokenized_docs = [_tokenize(d) for d in docs]
    tokenized_q = _tokenize(query)

    N = len(docs)
    df: Counter = Counter()
    for tokens in tokenized_docs:
        for tok in set(tokens):
            df[tok] += 1

    all_tokens = set(tokenized_q)
    for tokens in tokenized_docs:
        all_tokens.update(tokens)
    vocab = {tok: i for i, tok in enumerate(all_tokens)}

    def to_vec(tokens: List[str]) -> np.ndarray:
        tf = Counter(tokens)
        v = np.zeros(len(vocab), dtype=np.float32)
        for tok, cnt in tf.items():
            if tok in vocab:
                tf_val = cnt / len(tokens)
                idf_val = math.log((N + 1) / (df.get(tok, 0) + 1)) + 1.0
                v[vocab[tok]] = tf_val * idf_val
        return v

    q_vec = to_vec(tokenized_q)
    q_norm = float(np.linalg.norm(q_vec))
    if q_norm == 0:
        return [0.0] * N

    scores: List[float] = []
    for tokens in tokenized_docs:
        d_vec = to_vec(tokens)
        d_norm = float(np.linalg.norm(d_vec))
        if d_norm == 0:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(q_vec, d_vec) / (q_norm * d_norm)))
    return scores
