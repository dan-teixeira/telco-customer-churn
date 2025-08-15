"""
    Utility functions to simplify data wrangling and DataFrame manipulation.
"""

import unicodedata
import pandas as pd
from numpy import vectorize
import re


def vector(f):
    def wrapper(*arg, **kwargs):
        return vectorize(f)(*arg, **kwargs)

    return wrapper


@vector
def normalize_string(string: str) -> str:
    return "".join(
        [
            s
            for s in unicodedata.normalize(
                "NFD",
                re.sub(
                    string=string.strip().lower().replace(" ", "_"),
                    pattern=r"[^a-zA-Z1-9\s_]",
                    repl="",
                ),
            )
            if unicodedata.category(s) != "Mn"
        ]
    )


@vector
def pascal_to_snake(string: str) -> str:

    return (
        string[0] + "".join([f"_{s}" if s.isupper() else s for s in string[1:]])
    ).lower()


def get_formula(df: pd.DataFrame, endog: str, drop_columns: list = []) -> str:

    return (
        endog
        + " ~ "
        + " + ".join(
            [col for col in df.drop(drop_columns, axis=1).columns if col != endog]
        )
    )
