"""Baseline modeller; her model predict(df) ile satır başına 25 MisconceptionId döndürür."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PopularityBaseline:
    """Her satır için train'deki en sık k kavram yanılgısını tahmin eder."""

    def __init__(self, k: int = 25) -> None:
        self.k = k
        self.top_ids_: list[int] | None = None

    def fit(self, df: pd.DataFrame) -> "PopularityBaseline":
        counts = df["MisconceptionId"].dropna().astype("int64").value_counts()
        ranked = counts.reset_index()
        ranked.columns = ["MisconceptionId", "count"]
        ranked = ranked.sort_values(
            ["count", "MisconceptionId"], ascending=[False, True]
        )
        self.top_ids_ = ranked["MisconceptionId"].head(self.k).astype(int).tolist()
        return self

    def predict(self, df: pd.DataFrame) -> list[list[int]]:
        if self.top_ids_ is None:
            raise RuntimeError("Önce fit() çağrılmalı.")
        return [list(self.top_ids_) for _ in range(len(df))]


class TfidfCosineBaseline:
    """QueryText ile kavram yanılgılarını TF-IDF kosinüs benzerliğine göre sıralar."""

    def __init__(
        self,
        k: int = 25,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int | None = 50000,
    ) -> None:
        self.k = k
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer_: TfidfVectorizer | None = None
        self.candidate_matrix_ = None
        self.candidate_ids_: np.ndarray | None = None

    def fit(
        self,
        misconceptions: pd.DataFrame,
        train_df: pd.DataFrame | None = None,
    ) -> "TfidfCosineBaseline":
        cand_texts = (
            misconceptions["MisconceptionName"].fillna("").astype(str).tolist()
        )
        corpus = list(cand_texts)
        if train_df is not None and "QueryText" in train_df.columns:
            corpus.extend(train_df["QueryText"].fillna("").astype(str).tolist())

        self.vectorizer_ = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words="english",
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b|[%^/=<>]",
        )
        self.vectorizer_.fit(corpus)
        self.candidate_matrix_ = self.vectorizer_.transform(cand_texts)
        self.candidate_ids_ = (
            misconceptions["MisconceptionId"].astype(int).to_numpy()
        )
        return self

    def predict(self, df: pd.DataFrame) -> list[list[int]]:
        if (
            self.vectorizer_ is None
            or self.candidate_matrix_ is None
            or self.candidate_ids_ is None
        ):
            raise RuntimeError("Önce fit() çağrılmalı.")

        queries = df["QueryText"].fillna("").astype(str).tolist()
        scores = cosine_similarity(
            self.vectorizer_.transform(queries), self.candidate_matrix_
        )
        top_indices = np.argsort(-scores, axis=1)[:, : self.k]
        return [self.candidate_ids_[row_idx].tolist() for row_idx in top_indices]


class EmbeddingCosineBaseline:
    """QueryText ile kavram yanılgılarını cümle gömmesi kosinüs benzerliğine göre sıralar."""

    def __init__(
        self,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        k: int = 25,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.k = k
        self.batch_size = batch_size
        self.device = device
        self.encoder_ = None
        self.candidate_embeddings_: np.ndarray | None = None
        self.candidate_ids_: np.ndarray | None = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.encoder_.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def fit(
        self,
        misconceptions: pd.DataFrame,
        train_df: pd.DataFrame | None = None,
    ) -> "EmbeddingCosineBaseline":
        from sentence_transformers import SentenceTransformer

        self.encoder_ = SentenceTransformer(
            self.model_name_or_path, device=self.device
        )
        cand_texts = (
            misconceptions["MisconceptionName"].fillna("").astype(str).tolist()
        )
        self.candidate_embeddings_ = self._encode(cand_texts)
        self.candidate_ids_ = (
            misconceptions["MisconceptionId"].astype(int).to_numpy()
        )
        return self

    def predict(self, df: pd.DataFrame) -> list[list[int]]:
        if (
            self.encoder_ is None
            or self.candidate_embeddings_ is None
            or self.candidate_ids_ is None
        ):
            raise RuntimeError("Önce fit() çağrılmalı.")

        queries = df["QueryText"].fillna("").astype(str).tolist()
        scores = self._encode(queries) @ self.candidate_embeddings_.T
        top_indices = np.argsort(-scores, axis=1)[:, : self.k]
        return [self.candidate_ids_[row_idx].tolist() for row_idx in top_indices]
