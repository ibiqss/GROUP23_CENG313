"""Değerlendirme metriği, doğrulama ayrımı ve submission yardımcıları."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def map_at_k(
    y_true: Iterable[int],
    y_pred: Iterable[Sequence[int]],
    k: int = 25,
) -> float:
    """Tek etiketli hedefler için MAP@k hesaplar."""
    scores: list[float] = []
    for target, preds in zip(y_true, y_pred):
        target_int = int(target)
        unique_preds = list(dict.fromkeys(int(p) for p in preds))[:k]
        if target_int in unique_preds:
            scores.append(1.0 / (unique_preds.index(target_int) + 1))
        else:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def make_holdout_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = "QuestionId",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Soruların %20'sini ayırır; aynı sorunun şıkları aynı tarafta kalır."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, val_idx = next(splitter.split(df, groups=df[group_col]))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )


def make_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "QuestionId",
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """QuestionId'ye göre gruplanmış k katlı train/validation ayrımı üretir."""
    gkf = GroupKFold(n_splits=n_splits)
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_idx, val_idx in gkf.split(df, groups=df[group_col]):
        splits.append(
            (
                df.iloc[train_idx].reset_index(drop=True),
                df.iloc[val_idx].reset_index(drop=True),
            )
        )
    return splits


def evaluate_predictions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_pred: Sequence[Sequence[int]],
    k: int = 25,
) -> dict[str, float]:
    """Genel, train'de görülen ve görülmeyen etiketler için ayrı MAP@k döndürür."""
    if len(val_df) != len(y_pred):
        raise ValueError("val_df ve y_pred uzunlukları eşit olmalı.")
    y_true = val_df["MisconceptionId"].astype(int).to_numpy()
    seen_ids = set(train_df["MisconceptionId"].dropna().astype(int).unique())
    is_seen = np.isin(y_true, list(seen_ids))

    def _score(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        preds = [pred for pred, flag in zip(y_pred, mask) if flag]
        return map_at_k(y_true[mask], preds, k=k)

    everything = np.ones(len(y_true), dtype=bool)
    return {
        f"MAP@{k} (Overall)": _score(everything),
        f"MAP@{k} (Seen)": _score(is_seen),
        f"MAP@{k} (Unseen)": _score(~is_seen),
        "n_seen": int(is_seen.sum()),
        "n_unseen": int((~is_seen).sum()),
    }


def write_submission(
    df: pd.DataFrame,
    predictions: Sequence[Sequence[int]],
    output_path: str | Path = "submission.csv",
    k: int = 25,
) -> pd.DataFrame:
    """Tahminleri Kaggle formatında (QuestionId_Answer) CSV olarak kaydeder."""
    if len(df) != len(predictions):
        raise ValueError("df ve predictions uzunlukları eşit olmalı.")

    submission = pd.DataFrame(
        {
            "QuestionId_Answer": [
                f"{qid}_{ans}" for qid, ans in zip(df["QuestionId"], df["Answer"])
            ],
            "MisconceptionId": [
                " ".join(str(int(x)) for x in list(dict.fromkeys(preds))[:k])
                for preds in predictions
            ],
        }
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)
    return submission
