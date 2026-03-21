"""
轻量向量存储 —— 用 SQLite + numpy TF-IDF 替代 chromadb
实现与 chromadb 相同的接口，memory_hub.py 无需大改
"""
import os
import math
import json
import sqlite3
from collections import Counter
from typing import List, Dict, Optional

import numpy as np


class Collection:
    def __init__(self, db_path: str, name: str):
        self.name = name
        self.db_path = db_path
        self._init_table()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        with self._conn() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS "{self.name}" (
                    id       TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{{}}'
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------ write

    def add(self, documents: List[str], metadatas: List[dict], ids: List[str]):
        with self._conn() as conn:
            conn.executemany(
                f'INSERT OR REPLACE INTO "{self.name}" (id, document, metadata) VALUES (?, ?, ?)',
                [(i, d, json.dumps(m, ensure_ascii=False))
                 for i, d, m in zip(ids, documents, metadatas)],
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
                f'SELECT id, document, metadata FROM "{self.name}"'
            ).fetchall()
        return {
            "ids":       [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [json.loads(r[2]) for r in rows],
        }

    def query(self, query_texts: List[str], n_results: int = 5) -> dict:
        data = self.get()
        docs = data["documents"]
        if not docs:
            return {"documents": [[]], "metadatas": [[]]}

        scores = _tfidf_cosine(query_texts[0], docs)
        n = min(n_results, len(docs))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

        return {
            "documents": [[docs[i] for i in top]],
            "metadatas": [[data["metadatas"][i] for i in top]],
        }


# ------------------------------------------------------------------ client

class PersistentClient:
    def __init__(self, path: str):
        os.makedirs(path, exist_ok=True)
        self._db = os.path.join(path, "store.db")
        self._cols: Dict[str, Collection] = {}

    def get_or_create_collection(self, name: str, metadata: dict = None) -> Collection:
        if name not in self._cols:
            self._cols[name] = Collection(self._db, name)
        return self._cols[name]


# ------------------------------------------------------------------ similarity

def _tokenize(text: str) -> List[str]:
    """汉字 unigram + bigram，兼顾单字词和双字词语义"""
    tokens: List[str] = []
    for i, ch in enumerate(text):
        tokens.append(ch)
        if i < len(text) - 1:
            tokens.append(text[i: i + 2])
    return tokens or [""]


def _tfidf_cosine(query: str, docs: List[str]) -> List[float]:
    tokenized_docs = [_tokenize(d) for d in docs]
    tokenized_q = _tokenize(query)

    # 用文档集合计算 IDF（不含 query）
    N = len(docs)
    df: Counter = Counter()
    for tokens in tokenized_docs:
        for tok in set(tokens):
            df[tok] += 1

    # 收集词表
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
