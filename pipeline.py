"""Executa a preparação, um resumo de elasticidade e a modelagem."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

if __package__:
    from .data_utils import load_sales_data, prepare_model_data
    from .model import run_model
else:
    from data_utils import load_sales_data, prepare_model_data
    from model import run_model


def elasticidade_por_sku(group: pd.DataFrame) -> pd.Series:
    sample = group.loc[
        (group["qt_ruptura"] == 0)
        & (group["demanda_real"] > 0)
        & (group["preco_medio"] > 0)
    ].copy()
    if len(sample) < 15:
        return pd.Series({"elast": np.nan, "pval": np.nan})

    sample["ln_q"] = np.log(sample["demanda_real"])
    sample["ln_p"] = np.log(sample["preco_medio"])
    fitted_model = smf.ols(
        "ln_q ~ ln_p + flag_desconto_direto + flag_combo + "
        "flag_pague_e_leve + C(mes)",
        data=sample,
    ).fit()
    return pd.Series(
        {
            "elast": fitted_model.params["ln_p"],
            "pval": fitted_model.pvalues["ln_p"],
        }
    )


def main() -> None:
    df = prepare_model_data(load_sales_data())
    print(
        df.groupby("canal")[
            ["demanda_real", "preco_medio", "desconto_pct"]
        ].mean()
    )
    elasticity = (
        df.groupby("sku_id")
        .apply(elasticidade_por_sku)
        .sort_values("elast")
    )
    print(elasticity)
    run_model(df)


if __name__ == "__main__":
    main()
