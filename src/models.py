"""Baseline modeller. Her model ``predict(df)`` metodunu sağlar.

``predict``, ``df`` içindeki her satır için en olasıdan en az olasıya sıralanmış
en fazla ``k`` MisconceptionId içeren bir liste döndürür.
"""

from __future__ import annotations

import pandas as pd


class PopularityBaseline:
    """Her satır için eğitimdeki en sık misconception'ları tahmin eder.

    Soru metnini hiç kullanmaz; bu yüzden metin kullanan her modelin geçmesi
    gereken alt sınırı belirler.
    """

    def __init__(self, k: int = 25) -> None:
        self.k = k
        self.top_ids_: list[int] | None = None

    def fit(self, df: pd.DataFrame) -> "PopularityBaseline":
        counts = df["MisconceptionId"].dropna().astype("int64").value_counts()
        # Eşit sayılar id'ye göre sıralanır; böylece her çalıştırmada aynı tahmin çıkar.
        ranked = counts.reset_index()
        ranked.columns = ["MisconceptionId", "count"]
        ranked = ranked.sort_values(
            ["count", "MisconceptionId"], ascending=[False, True]
        )
        self.top_ids_ = ranked["MisconceptionId"].head(self.k).astype(int).tolist()
        return self

    def predict(self, df: pd.DataFrame) -> list[list[int]]:
        if self.top_ids_ is None:
            raise RuntimeError("predict() öncesinde fit() çağrılmalı.")
        return [list(self.top_ids_) for _ in range(len(df))]
