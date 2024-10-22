"""fugashi_utils module.

Copyright (C) 2024 Takaaki Shiotani

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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
