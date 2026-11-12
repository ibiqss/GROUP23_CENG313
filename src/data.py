"""Eedi yarışma dosyalarını yükler ve modellemeye uygun hale getirir.

Ham dosyalarda her satır bir sorudur. Projedeki tüm modeller ise her yanlış şık
(distractor) için bir satırla çalışır; ``load_long`` bu tabloyu döndürür:

    QuestionId | Answer | QueryText | MisconceptionId
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

COMPETITION = "eedi-mining-misconceptions-in-mathematics"

OPTIONS = ("A", "B", "C", "D")
ANSWER_TEXT_COLUMNS = [f"Answer{option}Text" for option in OPTIONS]
MISCONCEPTION_COLUMNS = [f"Misconception{option}Id" for option in OPTIONS]

QUESTION_COLUMNS = [
    "QuestionId",
    "ConstructId",
    "ConstructName",
    "SubjectId",
    "SubjectName",
    "CorrectAnswer",
    "QuestionText",
    *ANSWER_TEXT_COLUMNS,
]
MAPPING_COLUMNS = ["MisconceptionId", "MisconceptionName"]
LONG_COLUMNS = ["QuestionId", "Answer", "QueryText", "MisconceptionId"]

SPLITS = ("train", "test")


def _read_csv(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} bulunamadı. '{COMPETITION}' dosyalarını Kaggle'dan indirip "
            "veri klasörüne çıkarın."
        )
    frame = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} dosyasında beklenen sütunlar eksik: {missing}")
    return frame


def load_questions(data_dir: str | Path, split: str = "train") -> pd.DataFrame:
    """Ham ``train.csv`` veya ``test.csv`` tablosunu döndürür (soru başına bir satır)."""
    if split not in SPLITS:
        raise ValueError(f"split şunlardan biri olmalı: {SPLITS}, verilen: {split!r}")
    required = list(QUESTION_COLUMNS)
    if split == "train":
        required += MISCONCEPTION_COLUMNS
    return _read_csv(Path(data_dir) / f"{split}.csv", required)


def load_misconceptions(data_dir: str | Path) -> pd.DataFrame:
    """``misconception_mapping.csv`` dosyasını, yani modellerin sıraladığı adayları döndürür."""
    return _read_csv(Path(data_dir) / "misconception_mapping.csv", MAPPING_COLUMNS)


def clean_text(value: object) -> str:
    """Metnin başındaki ve sonundaki boşlukları siler, ardışık boşlukları teke indirir.

    Eksik değer boş metne dönüşür.
    """
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_query_text(question: object, correct: object, incorrect: object) -> str:
    """Soruyu, doğru cevabı ve bir yanlış cevabı tek metinde birleştirir."""
    return (
        f"Question: {clean_text(question)}\n"
        f"Correct answer: {clean_text(correct)}\n"
        f"Incorrect answer: {clean_text(incorrect)}"
    )


def to_long(
    questions: pd.DataFrame,
    drop_unlabeled: bool = True,
    keep_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Soru başına bir satırı, yanlış şık başına bir satıra çevirir.

    Doğru şık her zaman çıkarılır. ``keep_columns`` ile soruya ait ek sütunlar
    (ör. ``SubjectName``) dört standart sütunun yanına kopyalanır.
    Etiket içermeyen test verisinde ``drop_unlabeled`` etkisizdir.
    """
    extra_columns = list(keep_columns)
    has_labels = set(MISCONCEPTION_COLUMNS) <= set(questions.columns)

    invalid = ~questions["CorrectAnswer"].isin(OPTIONS)
    if invalid.any():
        bad_ids = questions.loc[invalid, "QuestionId"].tolist()
        raise ValueError(f"CorrectAnswer {OPTIONS} dışında olan sorular: {bad_ids}")

    rows = []
    for question in questions.to_dict("records"):
        correct_option = question["CorrectAnswer"]
        correct_text = question[f"Answer{correct_option}Text"]
        for option in OPTIONS:
            if option == correct_option:
                continue
            rows.append(
                {
                    "QuestionId": question["QuestionId"],
                    "Answer": option,
                    "QueryText": build_query_text(
                        question["QuestionText"],
                        correct_text,
                        question[f"Answer{option}Text"],
                    ),
                    "MisconceptionId": (
                        question[f"Misconception{option}Id"] if has_labels else pd.NA
                    ),
                    **{column: question[column] for column in extra_columns},
                }
            )

    long = pd.DataFrame(rows, columns=[*LONG_COLUMNS, *extra_columns])
    long["MisconceptionId"] = (
        pd.to_numeric(long["MisconceptionId"], errors="coerce").astype("Int64")
    )
    if drop_unlabeled and has_labels:
        long = long.dropna(subset=["MisconceptionId"]).reset_index(drop=True)
        long["MisconceptionId"] = long["MisconceptionId"].astype("int64")
    return long


def load_long(
    data_dir: str | Path,
    split: str = "train",
    drop_unlabeled: bool = True,
) -> pd.DataFrame:
    """Bir veri bölümünü, modellere hazır şekilde yanlış şık başına bir satır olarak yükler."""
    return to_long(load_questions(data_dir, split), drop_unlabeled=drop_unlabeled)
