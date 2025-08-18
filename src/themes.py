from __future__ import annotations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import re

STOPWORDS = None  # rely on built-in english stopwords in TfidfVectorizer


def _prep_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True)


def extract_themes(df: pd.DataFrame, text_col: str, group_cols: list[str] | None = None, top_n: int = 10) -> pd.DataFrame:
    group_cols = group_cols or []
    texts = df.copy()
    texts[text_col] = _prep_text(texts[text_col])

    if not group_cols:
        return _themes_for_group(texts[text_col], top_n).assign(**{"_scope": "all"})

    out = []
    for keys, grp in texts.groupby(group_cols):
        res = _themes_for_group(grp[text_col], top_n)
        if not isinstance(keys, tuple):
            keys = (keys,)
        for i, k in enumerate(group_cols):
            res[k] = keys[i]
        out.append(res)
    return pd.concat(out, ignore_index=True)


def _themes_for_group(texts: pd.Series, top_n: int) -> pd.DataFrame:
    if len(texts) == 0:
        return pd.DataFrame({"term": [], "score": []})
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts.tolist())
    scores = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    order = scores.argsort()[::-1][:top_n]
    return pd.DataFrame({"term": terms[order], "score": scores[order]})