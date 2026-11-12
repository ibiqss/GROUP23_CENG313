"""Yardımcı fonksiyonlar: veri kalitesi kontrolleri, EDA istatistikleri ve grafikler."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ANSWER_TEXT_COLUMNS, OPTIONS, clean_text

SERIES_COLOR = "#2a78d6"
SURFACE_COLOR = "#fcfcfb"
TEXT_COLOR = "#0b0b0b"
MUTED_TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_QUERY_TEMPLATE_WORDS = {"question", "correct", "incorrect", "answer"}


def apply_plot_style() -> None:
    """Notebook boyunca tek ve sade bir grafik stili kullanır."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE_COLOR,
            "axes.facecolor": SURFACE_COLOR,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": MUTED_TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED_TEXT_COLOR,
            "ytick.color": MUTED_TEXT_COLOR,
            "figure.dpi": 110,
        }
    )


# --------------------------------------------------------------------------
# Veri kalitesi
# --------------------------------------------------------------------------


def missing_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Sütun başına eksik değer sayısını ve yüzdesini, en kötüden başlayarak verir."""
    missing = frame.isna().sum()
    table = pd.DataFrame(
        {"eksik": missing, "yüzde": (missing / len(frame) * 100).round(2)}
    )
    return table.sort_values("eksik", ascending=False)


def repeated_option_mask(questions: pd.DataFrame) -> pd.Series:
    """İki şıkkı aynı metne sahip soruları işaretler."""
    return questions[ANSWER_TEXT_COLUMNS].apply(
        lambda row: len({clean_text(value) for value in row}) < len(OPTIONS), axis=1
    )


def repeated_question_texts(questions: pd.DataFrame) -> pd.DataFrame:
    """Birden fazla QuestionId tarafından kullanılan her soru metnini özetler."""
    texts = questions["QuestionText"].map(clean_text)
    options = questions[ANSWER_TEXT_COLUMNS].map(clean_text).agg(" | ".join, axis=1)
    groups = [
        {
            "QuestionText": text,
            "soru_sayısı": len(group),
            "construct_sayısı": group["ConstructId"].nunique(),
            "şıklar_aynı": options[group.index].nunique() == 1,
        }
        for text, group in questions[texts.duplicated(keep=False)].groupby(texts)
    ]
    return pd.DataFrame(
        groups,
        columns=["QuestionText", "soru_sayısı", "construct_sayısı", "şıklar_aynı"],
    )


def quality_checks(
    questions: pd.DataFrame,
    misconceptions: pd.DataFrame,
    long_all: pd.DataFrame,
) -> pd.DataFrame:
    """Eğitim verisi üzerinde bütünlük kontrollerini çalıştırır.

    ``long_all`` etiketsiz satırları da içeren uzun tablodur. Sayının sıfır
    olması kontrolün geçtiği anlamına gelir; gürültü türleri ders sınıflandırmasını
    izler.
    """
    labeled = long_all.dropna(subset=["MisconceptionId"])
    known_ids = set(misconceptions["MisconceptionId"])
    label_columns = [f"Misconception{option}Id" for option in OPTIONS]

    correct_option_labeled = sum(
        int(
            questions.loc[
                questions["CorrectAnswer"] == option, f"Misconception{option}Id"
            ]
            .notna()
            .sum()
        )
        for option in OPTIONS
    )
    answers = questions[ANSWER_TEXT_COLUMNS]
    text_series = [
        *(questions[column] for column in ["QuestionText", *ANSWER_TEXT_COLUMNS]),
        misconceptions["MisconceptionName"],
    ]
    untidy_text = sum(
        int((series.dropna() != series.dropna().map(clean_text)).sum())
        for series in text_series
    )
    labels_per_query = labeled.groupby("QueryText")["MisconceptionId"].nunique()
    all_labels = questions[label_columns].stack()

    checks = [
        ("Tamamen aynı satır", "Tekrar", questions.duplicated().sum()),
        ("Tekrarlanan QuestionId", "Tekrar", questions["QuestionId"].duplicated().sum()),
        (
            "Daha önce geçen bir QuestionText'i tekrarlayan soru",
            "Tekrar",
            questions["QuestionText"].map(clean_text).duplicated().sum(),
        ),
        (
            "Listede tekrarlanan MisconceptionId",
            "Tekrar",
            misconceptions["MisconceptionId"].duplicated().sum(),
        ),
        (
            "Listede tekrarlanan MisconceptionName",
            "Tekrar",
            misconceptions["MisconceptionName"].map(clean_text).duplicated().sum(),
        ),
        (
            "A-D dışında CorrectAnswer",
            "Hatalı",
            (~questions["CorrectAnswer"].isin(OPTIONS)).sum(),
        ),
        ("Misconception etiketi taşıyan doğru şık", "Hatalı", correct_option_labeled),
        (
            "Listede olmayan etiket",
            "Hatalı",
            (~all_labels.astype("int64").isin(known_ids)).sum(),
        ),
        ("Soru metni eksik soru", "Eksik", questions["QuestionText"].isna().sum()),
        ("Eksik şık metni", "Eksik", answers.isna().sum().sum()),
        ("Aynı metne sahip şıkları olan soru", "Tutarsız", repeated_option_mask(questions).sum()),
        ("Fazladan boşluk içeren metin", "Tutarsız", untidy_text),
        (
            "Aynı sorguya farklı etiket",
            "Çelişkili",
            (labels_per_query > 1).sum(),
        ),
    ]
    table = pd.DataFrame(checks, columns=["kontrol", "gürültü_türü", "sayı"])
    table["sayı"] = table["sayı"].astype(int)
    return table


# --------------------------------------------------------------------------
# İstatistikler
# --------------------------------------------------------------------------


def word_count(texts: pd.Series) -> pd.Series:
    """Her metindeki boşlukla ayrılmış kelime sayısı."""
    return texts.fillna("").astype(str).str.split().str.len()


def has_latex(texts: pd.Series) -> pd.Series:
    """LaTeX içeren metinleri (ters eğik çizgi veya dolar işareti) işaretler."""
    return texts.fillna("").astype(str).str.contains(r"\\|\$", regex=True)


def has_image(texts: pd.Series) -> pd.Series:
    """Markdown alt metni olarak saklanan görselleri (``![...]``) işaretler."""
    return texts.fillna("").astype(str).str.contains("![", regex=False)


def misconception_frequency(
    labeled: pd.DataFrame, misconceptions: pd.DataFrame
) -> pd.DataFrame:
    """Hiç kullanılmayanlar dahil her adayın kaç kez etiket olduğunu sayar."""
    counts = labeled["MisconceptionId"].astype("int64").value_counts()
    table = misconceptions.copy()
    table["kullanım"] = table["MisconceptionId"].map(counts).fillna(0).astype(int)
    return table.sort_values(
        ["kullanım", "MisconceptionId"], ascending=[False, True]
    ).reset_index(drop=True)


def top_share(counts: pd.Series, top_n: int) -> float:
    """En sık ``top_n`` adayın tüm etiketler içindeki payı."""
    ordered = counts.sort_values(ascending=False)
    return float(ordered.head(top_n).sum() / ordered.sum())


def candidates_for_share(counts: pd.Series, share: float) -> int:
    """Etiketlerin ``share`` kadarını kapsamak için gereken en sık aday sayısı."""
    ordered = counts.sort_values(ascending=False).to_numpy()
    cumulative = np.cumsum(ordered) / ordered.sum()
    return int(np.searchsorted(cumulative, share) + 1)


def rank_correlation(left: pd.Series, right: pd.Series) -> float:
    """Spearman korelasyonu: sıraların Pearson korelasyonu."""
    return float(left.rank().corr(right.rank()))


def unseen_label_share(
    labeled: pd.DataFrame,
    holdout_fraction: float = 0.2,
    repeats: int = 20,
    seed: int = 42,
) -> pd.Series:
    """Ayrılan soruların etiketlerinden kaçının kalan veride hiç geçmediğini tahmin eder.

    Yeni test soruları gibi davranması için sorular bütün olarak ayrılır.
    """
    rng = np.random.default_rng(seed)
    question_ids = labeled["QuestionId"].unique()
    holdout_size = max(1, round(len(question_ids) * holdout_fraction))
    shares = []
    for _ in range(repeats):
        holdout_ids = rng.choice(question_ids, size=holdout_size, replace=False)
        in_holdout = labeled["QuestionId"].isin(holdout_ids)
        seen = set(labeled.loc[~in_holdout, "MisconceptionId"])
        unseen = ~labeled.loc[in_holdout, "MisconceptionId"].isin(seen)
        shares.append(float(unseen.mean()))
    return pd.Series(shares, name="görülmemiş_oranı")


def _content_tokens(text: object) -> set[str]:
    return set(_TOKEN_PATTERN.findall(str(text).lower())) - _QUERY_TEMPLATE_WORDS


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def lexical_overlap(
    queries: Iterable[str], names: Sequence[str], seed: int = 42
) -> pd.DataFrame:
    """Gerçek sorgu-isim çiftleriyle karıştırılmış çiftlerin kelime örtüşmesini karşılaştırır.

    Gerçek çiftler rastgele çiftlerden daha fazla örtüşmüyorsa, basit kelime
    eşleştirmenin kullanabileceği bir sinyal yok demektir.
    """
    query_tokens = [_content_tokens(text) for text in queries]
    name_tokens = [_content_tokens(text) for text in names]
    order = np.random.default_rng(seed).permutation(len(name_tokens))
    return pd.DataFrame(
        {
            "gerçek_çift": [
                _jaccard(query, name) for query, name in zip(query_tokens, name_tokens)
            ],
            "karıştırılmış_çift": [
                _jaccard(query, name_tokens[index])
                for query, index in zip(query_tokens, order)
            ],
        }
    )


def top_k_hit_rate(y_true: Iterable[int], y_pred: Iterable[Sequence[int]]) -> float:
    """Doğru etiketin tahmin listesinde herhangi bir sırada yer aldığı satırların oranı."""
    hits = [label in predictions for label, predictions in zip(y_true, y_pred)]
    return float(np.mean(hits))


# --------------------------------------------------------------------------
# Grafikler
# --------------------------------------------------------------------------


def _shorten(label: object, width: int = 60) -> str:
    text = str(label)
    return text if len(text) <= width else text[: width - 1] + "…"


def plot_bars(values: pd.Series, title: str, xlabel: str) -> plt.Axes:
    """Verilen sırada yatay çubuklar; her değer çubuğun ucunda yazar."""
    height = max(2.2, 0.32 * len(values) + 1.2)
    _, ax = plt.subplots(figsize=(9, height))
    positions = np.arange(len(values))
    ax.barh(
        positions,
        values.to_numpy(),
        height=0.72,
        color=SERIES_COLOR,
        edgecolor=SURFACE_COLOR,
        linewidth=1.5,
    )
    ax.set_yticks(positions, [_shorten(label) for label in values.index])
    ax.invert_yaxis()
    ax.grid(False, axis="y")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    labels = [f"{value:,.3g}" if value < 1 else f"{value:,.0f}" for value in values]
    ax.bar_label(ax.containers[0], labels=labels, padding=3, color=MUTED_TEXT_COLOR, fontsize=9)
    ax.margins(x=0.08)
    plt.tight_layout()
    plt.show()
    return ax


def plot_histogram(
    values: pd.Series, title: str, xlabel: str, bins: int = 40
) -> plt.Axes:
    """Medyanı işaretlenmiş histogram."""
    _, ax = plt.subplots(figsize=(9, 3.4))
    ax.hist(
        values,
        bins=bins,
        color=SERIES_COLOR,
        edgecolor=SURFACE_COLOR,
        linewidth=1.0,
    )
    median = float(values.median())
    ax.axvline(median, color=TEXT_COLOR, linewidth=1.5)
    ax.annotate(
        f"medyan {median:g}",
        xy=(median, 1),
        xycoords=("data", "axes fraction"),
        xytext=(5, -12),
        textcoords="offset points",
        color=TEXT_COLOR,
        fontsize=9,
    )
    ax.grid(False, axis="x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sayı")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return ax


def plot_rank_frequency(counts: pd.Series, title: str) -> plt.Axes:
    """Kullanılan adayların sıklığı sıraya göre, logaritmik eksende."""
    ordered = counts[counts > 0].sort_values(ascending=False).to_numpy()
    _, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(np.arange(1, len(ordered) + 1), ordered, color=SERIES_COLOR, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Misconception sırası (1 = en sık)")
    ax.set_ylabel("Etiketli satır (log ölçek)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return ax


def plot_cumulative_share(counts: pd.Series, title: str) -> plt.Axes:
    """En sık N adayın kapsadığı birikimli etiket payı."""
    ordered = counts.sort_values(ascending=False).to_numpy()
    share = np.cumsum(ordered) / ordered.sum()
    _, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(np.arange(1, len(share) + 1), share * 100, color=SERIES_COLOR, linewidth=2)
    ax.set_ylim(0, 102)
    ax.set_xlabel("En sık misconception sayısı")
    ax.set_ylabel("Etiketlerin payı (%)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return ax
