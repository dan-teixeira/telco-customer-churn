"""
    Functions to evaluate model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import add_constant
from sklearn.metrics import accuracy_score, recall_score
import warnings

warnings.filterwarnings(action="ignore", category=RuntimeWarning, module="statsmodels")


def vif_tolerance(df: pd.DataFrame, endog: str, drop_columns: list = []):

    df = add_constant(df.drop(drop_columns + [endog], axis=1))

    df_metrics = pd.DataFrame(
        zip(
            df.columns[1:],
            [
                variance_inflation_factor(exog=df.values, exog_idx=idx)
                for idx in range(1, df.shape[1])
            ],
        ),
        columns=["variables", "vif"],
    )

    df_metrics["tolerance"] = df_metrics["vif"] ** (-1)

    return df_metrics.sort_values("vif", ascending=False).reset_index(drop=True)


def confusion_matrix(logit_model, cutoff: int, color: str = "Blues"):

    classified = pd.Series(logit_model.predict() >= cutoff, name="Classified")

    observed = pd.Series(logit_model.model.endog.astype(bool), name="Observed")

    confusion_matrix = pd.crosstab(classified, observed)

    plt.figure(figsize=(8, 6), dpi=80)

    ax = sns.heatmap(
        data=confusion_matrix,
        cbar=False,
        annot=True,
        fmt="9.0f",
        cmap=color,
        annot_kws={"size": 18},
    )

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15, rotation=0)

    plt.xlabel(ax.get_xlabel(), fontsize=15)
    plt.ylabel(ax.get_ylabel(), fontsize=15)

    plt.title(f"cutoff = {cutoff:.2%}")
    plt.suptitle("Confusion Matrix", fontsize=18)

    return {
        "Accuracy": accuracy_score(y_true=observed, y_pred=classified),
        "Sensitivity": float(
            recall_score(y_true=observed, y_pred=classified, pos_label=1)
        ),
        "Specificity": float(
            recall_score(y_true=observed, y_pred=classified, pos_label=0)
        ),
    }


def confusion_metrics(logit_model) -> pd.DataFrame:

    observed = logit_model.model.endog

    predict = logit_model.predict()

    df = pd.DataFrame(
        columns=["accuracy", "sensitivity", "specificity"],
        index=pd.Series(name="cutoff"),
    )

    for cutoff in np.linspace(0.01, 1, 100):

        classified = predict >= cutoff

        df.loc[cutoff] = [
            accuracy_score(y_true=observed, y_pred=classified),
            recall_score(
                y_true=observed, y_pred=classified, pos_label=1, zero_division=np.nan
            ),
            recall_score(
                y_true=observed, y_pred=classified, pos_label=0, zero_division=np.nan
            ),
        ]

    return df.reset_index()
