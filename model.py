"""Treina, avalia e interpreta o modelo de demanda do case."""

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.metrics import mean_absolute_error

if __package__:
    from .data_utils import load_sales_data, prepare_model_data
else:
    from data_utils import load_sales_data, prepare_model_data


CUTOFF = pd.Timestamp("2026-04-30")
FEATURES = [
    "preco_medio",
    "desconto_pct",
    "preco_rel",
    "flag_desconto_direto",
    "flag_combo",
    "flag_pague_e_leve",
    "wk_sin",
    "wk_cos",
    "mes",
    "dez",
    "tendencia",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "roll_mean_4",
    "roll_mean_8",
    "roll_std_4",
    "ruptura_lag1",
    "ruptura_flag_lag1",
    "te_sku_id",
    "te_canal",
    "te_subcategoria_id",
]
TARGET = "demanda_real"


def run_model(df: pd.DataFrame) -> dict[str, object]:
    """Ajusta os modelos e retorna modelos, previsões e métricas."""
    train = df.loc[df["semana"] <= CUTOFF].copy()
    test = df.loc[df["semana"] > CUTOFF].copy()
    if train.empty or test.empty:
        raise ValueError(
            "O split temporal ficou vazio. Confira as datas do CSV e o cutoff "
            f"({CUTOFF.date()})."
        )

    for column in ["sku_id", "canal", "subcategoria_id"]:
        encoding = train.groupby(column)[TARGET].mean()
        train[f"te_{column}"] = train[column].map(encoding)
        test[f"te_{column}"] = test[column].map(encoding)

    train_m = train.dropna(subset=[TARGET]).copy()
    test_m = test.dropna(subset=[TARGET]).copy()
    if train_m.empty or test_m.empty:
        raise ValueError("Não há observações com alvo válido no treino ou no teste.")

    validation_mask = train_m["semana"] > (
        CUTOFF - pd.Timedelta(weeks=8)
    )
    train_inner = train_m.loc[~validation_mask]
    validation = train_m.loc[validation_mask]
    if train_inner.empty or validation.empty:
        raise ValueError(
            "Não foi possível separar treino e validação. "
            "São necessárias observações antes e durante as últimas 8 semanas."
        )

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    training_data = lgb.Dataset(
        train_inner[FEATURES], label=train_inner[TARGET]
    )
    validation_data = lgb.Dataset(
        validation[FEATURES], label=validation[TARGET]
    )

    model = lgb.train(
        params,
        training_data,
        num_boost_round=2000,
        valid_sets=[validation_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    def train_quantile(alpha: float) -> lgb.Booster:
        quantile_params = params.copy()
        quantile_params.update(
            objective="quantile", alpha=alpha, metric="quantile"
        )
        return lgb.train(
            quantile_params,
            training_data,
            num_boost_round=2000,
            valid_sets=[validation_data],
            callbacks=[lgb.early_stopping(100)],
        )

    model_lo = train_quantile(0.1)
    model_hi = train_quantile(0.9)

    test_m["pred"] = np.clip(model.predict(test_m[FEATURES]), 0, None)
    test_m["pred_lo"] = np.clip(model_lo.predict(test_m[FEATURES]), 0, None)
    test_m["pred_hi"] = np.clip(model_hi.predict(test_m[FEATURES]), 0, None)
    test_m["pred_lo"] = np.minimum(test_m["pred_lo"], test_m["pred"])
    test_m["pred_hi"] = np.maximum(test_m["pred_hi"], test_m["pred"])

    actual = test_m[TARGET]
    denominator = np.abs(actual).sum()
    if denominator == 0:
        raise ValueError("WAPE indefinido: a demanda real do teste soma zero.")

    wape = np.abs(actual - test_m["pred"]).sum() / denominator
    mean_actual = actual.mean()
    bias = (
        (test_m["pred"] - actual).mean() / mean_actual
        if mean_actual != 0
        else np.nan
    )
    print(f"WAPE: {wape:.4f}")
    print(f"MAE : {mean_absolute_error(actual, test_m['pred']):.4f}")
    print(f"Bias: {bias:.4f}")

    interval_width = (
        test_m["pred_hi"] - test_m["pred_lo"]
    ) / test_m["pred"].replace(0, np.nan)
    test_m["confianca"] = interval_width.apply(
        lambda value: (
            "Alta"
            if value < 0.5
            else ("Média" if value < 1.0 else "Baixa")
        )
    )
    print(test_m["confianca"].value_counts(normalize=True))

    coverage = (
        (actual >= test_m["pred_lo"]) & (actual <= test_m["pred_hi"])
    ).mean()
    print(f"Cobertura 80%: {coverage:.2%}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_m[FEATURES].fillna(0))
    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=FEATURES
    ).sort_values(ascending=False)
    print(importance.head(15))

    return {
        "model": model,
        "model_lo": model_lo,
        "model_hi": model_hi,
        "predictions": test_m,
        "metrics": {"wape": wape, "mae": mean_absolute_error(actual, test_m["pred"]),
                    "bias": bias, "coverage_80": coverage},
        "feature_importance": importance,
    }


def main() -> None:
    df = prepare_model_data(load_sales_data())
    run_model(df)


if __name__ == "__main__":
    main()
