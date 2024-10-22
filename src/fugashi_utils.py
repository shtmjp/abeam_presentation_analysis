from __future__ import annotations

import pandas as pd
from fugashi import Tagger  # type: ignore[attr-defined]


def fugashi_verbose_to_df(s: str) -> pd.DataFrame:
    """Convert fugashi verbose output to DataFrame."""
    # description of columns can be found in lib/site-packages/unidic/dicdir/dicrc
    tagger = Tagger("-Overbose")
    f = tagger.parse(s)
    dict_lines = []
    for line in f.splitlines()[:-1]:  # avoid the last EOS line
        d = {}
        for feat_str in line.split("\t"):
            if feat_str == "":
                continue
            feat_name, value = feat_str.split(":")
            d[feat_name] = value
        dict_lines.append(d)
    return pd.DataFrame(dict_lines)


def get_yomi(txt: str) -> str:
    """Get yomi from text."""
    fuga_df = fugashi_verbose_to_df(txt)
    try:
        prons = fuga_df["pron"].fillna(fuga_df["surface"]).to_list()
    except KeyError:
        prons = fuga_df["surface"].to_list()
    return "".join(prons)
