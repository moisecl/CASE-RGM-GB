"""Treina, seleciona e avalia previsões recursivas de vendas para oito semanas.

LightGBM e XGBoost são comparados em quatro janelas temporais de validação,
cada uma simulando oito semanas à frente sem consultar as vendas reais dentro
do horizonte. Os lags e as médias móveis são atualizados com as próprias
previsões. Preço, promoção e ruptura futura não são usados, pois não há planos
futuros disponíveis para esses dados.

Após a seleção e o ajuste de parâmetros, o modelo é treinado com todo o período
anterior às oito semanas finais e avaliado nessas semanas, também de forma
recursiva. Assim, o teste representa uma única previsão de oito semanas, e não
oito previsões semanais atualizadas com vendas reais.

A incerteza é estimada por regressão quantílica condicional, aplicada somente
depois da seleção do melhor modelo-base. Três modelos são ajustados para os
quantis 0.05, 0.50 e 0.95, gerando previsões P05, P50 e P95. A mediana é
realimentada no histórico durante a previsão recursiva. A cobertura observada
na validação é reportada como diagnóstico, não como garantia.

Gráficos SHAP Waterfall explicam contribuições locais para uma previsão de
SKU_01 na primeira semana do teste, em cada canal. As contribuições das
categorias one-hot são agrupadas pela variável original e expressas em unidades
vendidas. São exemplos de previsões específicas, não explicações causais.

Modelos:
* LightGBM: rápido e eficiente em dados tabulares; modelos muito complexos
  podem sobreajustar e árvores não extrapolam tendências.
* XGBoost: boosting flexível, com regularização e bom desempenho tabular;
  pode exigir mais tempo/recursos e árvores também não extrapolam tendências.

Referências:
* LightGBM: https://lightgbm.readthedocs.io/en/stable/
* XGBoost: https://xgboost.readthedocs.io/en/stable/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

if __package__:
    from src.data_utils import (
        carregar_dados_vendas,
        preparar_dados_modelo,
        preparar_features_semana,
    )
else:
    from src.data_utils import (
        carregar_dados_vendas,
        preparar_dados_modelo,
        preparar_features_semana,
    )


TARGET = "qt_vendida"
CV_FOLDS = 4
FORECAST_HORIZON_WEEKS = 8
CV_WEEKS = CV_FOLDS * FORECAST_HORIZON_WEEKS
FOLD_WEEKS = FORECAST_HORIZON_WEEKS
TEST_WEEKS = 8
MIN_TRAIN_WEEKS = 8
RANDOM_STATE = 42
RANDOM_SEARCH_ITERATIONS = 12
PREDICTION_QUANTILES = (0.05, 0.50, 0.95)
MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "model"

NUMERIC_FEATURES = [
    "semana_seno",
    "semana_cosseno",
    "venda_defasada_1",
    "venda_defasada_2",
    "venda_defasada_4",
    "venda_defasada_8",
    "media_movel_4",
    "media_movel_8",
    "desvio_padrao_movel_4",
]
CATEGORICAL_FEATURES = ["sku_id", "canal", "subcategoria_id"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Dicionário para tornar a entrada do modelo compreensível e auditável.
FEATURE_DESCRIPTIONS = {
    "semana_seno": "Componente seno que representa a posição cíclica da semana no ano.",
    "semana_cosseno": "Componente cosseno que representa a posição cíclica da semana no ano.",
    "venda_defasada_1": "Quantidade vendida na semana anterior para o mesmo SKU e canal.",
    "venda_defasada_2": "Quantidade vendida duas semanas antes para o mesmo SKU e canal.",
    "venda_defasada_4": "Quantidade vendida quatro semanas antes para o mesmo SKU e canal.",
    "venda_defasada_8": "Quantidade vendida oito semanas antes para o mesmo SKU e canal.",
    "media_movel_4": "Média das quatro quantidades anteriores, sem incluir a semana prevista.",
    "media_movel_8": "Média das oito quantidades anteriores, sem incluir a semana prevista.",
    "desvio_padrao_movel_4": "Desvio padrão das vendas nas quatro semanas anteriores.",
    "sku_id": "Identificador do SKU, codificado como categoria.",
    "canal": "Canal de venda, codificado como categoria.",
    "subcategoria_id": "Identificador da subcategoria, codificado como categoria.",
}


def _make_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Codifica categorias e preserva NaN numéricos para os modelos de boosting."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
        sparse_threshold=0,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessor), ("model", estimator)]
    )


def _save_shap_waterfall(
    pipeline: Pipeline,
    explainer: shap.TreeExplainer,
    feature_row: pd.DataFrame,
    prediction_row: pd.Series,
    filename: str,
) -> Path:
    """Salva uma explicação SHAP local agrupada por variável de negócio."""
    preprocessor = pipeline.named_steps["preprocessing"]
    estimator = pipeline.named_steps["model"]
    transformed_features = preprocessor.transform(feature_row[FEATURES])
    shap_values = explainer(transformed_features)
    if shap_values.values.ndim != 2 or shap_values.values.shape[0] != 1:
        raise ValueError("A explicação SHAP deve corresponder a uma previsão.")

    encoded_values = np.asarray(shap_values.values[0], dtype=float)
    encoder = pipeline.named_steps["preprocessing"].named_transformers_[
        "categorical"
    ]
    categorical_labels = {
        "sku_id": "SKU",
        "canal": "Canal",
        "subcategoria_id": "Subcategoria",
    }
    offset = 0
    grouped_values = []
    feature_labels = []
    for feature, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        next_offset = offset + len(categories)
        category_value = feature_row.iloc[0][feature]
        grouped_values.append(float(encoded_values[offset:next_offset].sum()))
        feature_labels.append(
            f"{categorical_labels[feature]}: {category_value}"
        )
        offset = next_offset

    if len(encoded_values) != offset + len(NUMERIC_FEATURES):
        raise ValueError(
            "A quantidade de valores SHAP não corresponde às "
            "features numéricas e categóricas transformadas."
        )
    numeric_labels = {
        "semana_seno": "Sazonalidade anual (componente seno)",
        "semana_cosseno": "Sazonalidade anual (componente cosseno)",
        "venda_defasada_1": "Vendas na semana anterior (un.)",
        "venda_defasada_2": "Vendas há 2 semanas (un.)",
        "venda_defasada_4": "Vendas há 4 semanas (un.)",
        "venda_defasada_8": "Vendas há 8 semanas (un.)",
        "media_movel_4": "Média de vendas nas últimas 4 semanas (un.)",
        "media_movel_8": "Média de vendas nas últimas 8 semanas (un.)",
        "desvio_padrao_movel_4": (
            "Variação das vendas nas últimas 4 semanas (un.)"
        ),
    }
    for index, feature in enumerate(NUMERIC_FEATURES):
        value = feature_row.iloc[0][feature]
        formatted_value = (
            "indisponível"
            if pd.isna(value)
            else f"{float(value):.2f}".replace(".", ",")
        )
        grouped_values.append(float(encoded_values[offset + index]))
        feature_labels.append(
            f"{numeric_labels[feature]}: {formatted_value}"
        )

    base_value = float(np.asarray(shap_values.base_values).reshape(-1)[0])
    raw_prediction = float(estimator.predict(transformed_features)[0])
    explained_prediction = base_value + sum(grouped_values)
    if not np.isclose(explained_prediction, raw_prediction, rtol=1e-4, atol=1e-4):
        raise ValueError(
            "Os valores SHAP não recompõem a saída do modelo para a previsão."
        )

    explanation = shap.Explanation(
        values=np.asarray(grouped_values),
        base_values=base_value,
        feature_names=feature_labels,
    )
    channel = prediction_row["canal"]
    week = prediction_row["semana"]
    final_prediction = float(prediction_row["previsao"])
    title = (
        f"Por que o modelo previu {final_prediction:.1f} unidades?\n"
        f"{prediction_row['sku_id']} | canal {channel} | "
        f"semana {week:%Y-%m-%d} (horizonte "
        f"{int(prediction_row['horizonte'])})"
    )
    if not np.isclose(raw_prediction, final_prediction, rtol=1e-5, atol=1e-5):
        title += (
            f"\nSaída bruta do modelo: {raw_prediction:.1f} un.; "
            "a previsão final é limitada a zero"
        )

    shap.plots.waterfall(
        explanation,
        max_display=len(feature_labels),
        show=False,
    )
    figure = plt.gcf()
    figure.set_size_inches(13, 9)
    plt.title(title, fontsize=12, pad=16)
    output_path = MODEL_OUTPUT_DIR / filename
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calcula WAPE, MAE e viés percentual."""
    denominator = actual.abs().sum()
    if denominator == 0:
        raise ValueError("WAPE indefinido: a soma do target é zero.")

    mean_actual = actual.mean()
    bias = (
        float((predicted - actual.to_numpy()).mean() / mean_actual)
        if mean_actual != 0
        else float("nan")
    )
    return {
        "wape": float(
            np.abs(actual.to_numpy() - predicted).sum() / denominator
        ),
        "mae": float(mean_absolute_error(actual, predicted)),
        "bias": bias,
    }


def _save_test_forecast_charts(predictions: pd.DataFrame) -> dict[str, Path]:
    """Salva um gráfico com 20 painéis por canal, cada um para um SKU."""
    channels = {
        "Venda Direta": "venda_direta",
        "Loja": "loja",
    }
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_paths = {}
    for channel, filename in channels.items():
        channel_predictions = predictions.loc[
            predictions["canal"] == channel
        ].sort_values(["sku_id", "semana"])
        skus = channel_predictions["sku_id"].drop_duplicates().sort_values()
        if len(skus) != 20:
            raise ValueError(
                f"Esperados 20 SKU-canal no canal {channel}; "
                f"foram encontrados {len(skus)}."
            )

        figure, axes = plt.subplots(4, 5, figsize=(20, 13), sharex=True)
        for axis, sku in zip(axes.flat, skus):
            group = channel_predictions.loc[
                channel_predictions["sku_id"] == sku
            ].sort_values("semana")
            axis.plot(
                group["semana"],
                group[TARGET],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                label="Real",
            )
            axis.plot(
                group["semana"],
                group["previsao"],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                linestyle="--",
                label="Previsto",
            )
            axis.fill_between(
                group["semana"],
                group["previsao_p05"],
                group["previsao_p95"],
                alpha=0.15,
                label="Intervalo quantílico P05–P95",
            )
            axis.set_title(str(sku))
            axis.grid(alpha=0.25)
            axis.tick_params(axis="x", labelrotation=45, labelsize=7)
            axis.tick_params(axis="y", labelsize=8)

        figure.suptitle(
            f"Vendas reais vs. previstas no teste — canal {channel}"
        )
        figure.supxlabel("Semana")
        figure.supylabel("Unidades vendidas")
        legend_handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=3,
        )
        figure.tight_layout(rect=(0.03, 0.03, 0.97, 0.91))

        output_path = (
            MODEL_OUTPUT_DIR / f"real_vs_previsto_{filename}.png"
        )
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        chart_paths[channel] = output_path

    return chart_paths


def _print_sku_business_translation(predictions: pd.DataFrame) -> None:
    """Traduz o erro de SKU_01 em unidades e valor bruto de vendas aproximado."""
    sku_predictions = predictions.loc[predictions["sku_id"] == "SKU_01"]
    print("\nExemplo de tradução do erro para KPI de negócio (SKU_01):")
    if sku_predictions.empty:
        print("SKU_01 não aparece no período de teste; KPI de exemplo indisponível.")
        return

    for channel, group in sku_predictions.groupby("canal", sort=True):
        absolute_error = (group[TARGET] - group["previsao"]).abs()
        actual_volume = group[TARGET].abs().sum()
        wape = (
            absolute_error.sum() / actual_volume
            if actual_volume
            else float("nan")
        )
        mae_units = absolute_error.mean()
        priced_rows = group.loc[
            group["preco_medio"].notna()
            & np.isfinite(group["preco_medio"])
            & group["preco_medio"].ge(0)
        ]
        if priced_rows.empty:
            value_message = "valor em R$ não estimado: não há preço válido"
        else:
            gross_value_error = (
                (priced_rows[TARGET] - priced_rows["previsao"]).abs()
                * priced_rows["preco_medio"]
            ).mean()
            formatted_value = f"{gross_value_error:,.2f}".replace(",", "X")
            formatted_value = formatted_value.replace(".", ",").replace("X", ".")
            value_message = (
                f"valor bruto associado ao erro aprox. R$ {formatted_value}"
                + f"/semana, média de {len(priced_rows)} semanas com preço"
            )

        print(
            f"SKU_01 | canal {channel}: WAPE {wape:.1%}; erro absoluto médio "
            f"{mae_units:.1f} un./semana; {value_message}. "
            "É uma proxy sobre vendas observadas, não o custo financeiro "
            "efetivo de excesso de estoque ou ruptura."
        )


def _period_description(frame: pd.DataFrame) -> str:
    """Descreve as datas cobertas, semanas e linhas de um período."""
    return (
        f"{frame['semana'].min():%Y-%m-%d} a "
        f"{frame['semana'].max():%Y-%m-%d} "
        f"({frame['semana'].nunique()} semanas, {len(frame)} linhas)"
    )


def _make_temporal_folds(
    frame: pd.DataFrame, weeks: pd.Index
) -> list[tuple[np.ndarray, pd.Index]]:
    """Cria quatro origens de validação, cada uma com horizonte de oito semanas."""
    if FOLD_WEEKS <= 0 or CV_WEEKS % FOLD_WEEKS:
        raise ValueError("CV_WEEKS deve ser divisível por FOLD_WEEKS.")

    all_weeks = pd.Index(frame["semana"].drop_duplicates().sort_values())
    weeks = pd.Index(weeks)
    if len(weeks) != CV_WEEKS or not weeks.isin(all_weeks).all():
        raise ValueError(
            f"A validação deve conter exatamente {CV_WEEKS} semanas presentes "
            "no conjunto de dados."
        )

    folds = []
    for start in range(0, len(weeks), FOLD_WEEKS):
        validation_dates = weeks[start : start + FOLD_WEEKS]
        train_dates = all_weeks[all_weeks < validation_dates.min()]
        train_indices = np.flatnonzero(frame["semana"].isin(train_dates))
        if not len(train_indices):
            raise ValueError("O fold temporal não contém semanas de treino.")
        if train_dates.max() >= validation_dates.min():
            raise ValueError("O fold temporal inclui treino posterior à validação.")
        if len(validation_dates) != FOLD_WEEKS:
            raise ValueError(
                f"O fold não contém exatamente {FOLD_WEEKS} semanas."
            )
        folds.append((train_indices, validation_dates))
    return folds


def _forecast_recursively(
    pipeline: Pipeline,
    history: pd.DataFrame,
    forecast_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prevê semanas futuras em sequência, usando previsões como histórico."""
    forecast_weeks = pd.Index(
        forecast_rows["semana"].drop_duplicates().sort_values()
    )
    if len(forecast_weeks) != FORECAST_HORIZON_WEEKS:
        raise ValueError(
            "A previsão recursiva deve conter exatamente "
            f"{FORECAST_HORIZON_WEEKS} semanas."
        )
    if not (
        forecast_weeks.to_series().diff().dropna().dt.days.eq(7).all()
    ):
        raise ValueError("O horizonte de previsão deve conter semanas contíguas.")

    working_history = history[["sku_id", "canal", "semana", TARGET]].copy()
    results = []
    feature_rows = []
    for horizon, week in enumerate(forecast_weeks, 1):
        week_rows = forecast_rows.loc[forecast_rows["semana"] == week].copy()
        week_features = preparar_features_semana(working_history, week_rows)
        feature_rows.append(week_features[FEATURES].copy())
        prediction = np.clip(
            pipeline.predict(week_features[FEATURES]), 0, None
        )
        result_columns = ["semana", "sku_id", "canal"]
        if TARGET in week_rows:
            result_columns.append(TARGET)
        if "preco_medio" in week_rows:
            result_columns.append("preco_medio")
        week_result = week_rows[result_columns].copy()
        week_result["previsao"] = prediction
        week_result["horizonte"] = horizon
        results.append(week_result)

        predicted_history = week_rows[["sku_id", "canal", "semana"]].copy()
        predicted_history[TARGET] = prediction
        working_history = pd.concat(
            [working_history, predicted_history], ignore_index=True
        )

    forecast = pd.concat(results).sort_values(["semana", "sku_id", "canal"])
    forecast_features = pd.concat(feature_rows).loc[forecast.index]
    return forecast, forecast_features


def _make_quantile_estimators(
    model_name: str, estimator: BaseEstimator
) -> dict[float, BaseEstimator]:
    """Cria um estimador por quantil mantendo a configuração selecionada."""
    objective_parameters = {
        "LightGBM": ("quantile", "alpha"),
        "XGBoost": ("reg:quantileerror", "quantile_alpha"),
    }
    if model_name not in objective_parameters:
        raise ValueError(f"Regressão quantílica não configurada para {model_name}.")

    objective, quantile_parameter = objective_parameters[model_name]
    return {
        quantile: clone(estimator).set_params(
            objective=objective, **{quantile_parameter: quantile}
        )
        for quantile in PREDICTION_QUANTILES
    }


def _forecast_recursively_quantiles(
    pipelines: dict[float, Pipeline],
    history: pd.DataFrame,
    forecast_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prevê os quantis recursivamente, usando P50 como histórico futuro."""
    forecast_weeks = pd.Index(
        forecast_rows["semana"].drop_duplicates().sort_values()
    )
    if len(forecast_weeks) != FORECAST_HORIZON_WEEKS:
        raise ValueError(
            "A previsão recursiva deve conter exatamente "
            f"{FORECAST_HORIZON_WEEKS} semanas."
        )
    if not (
        forecast_weeks.to_series().diff().dropna().dt.days.eq(7).all()
    ):
        raise ValueError("O horizonte de previsão deve conter semanas contíguas.")
    if set(pipelines) != set(PREDICTION_QUANTILES):
        raise ValueError("É necessário fornecer os três modelos quantílicos.")

    working_history = history[["sku_id", "canal", "semana", TARGET]].copy()
    results = []
    feature_rows = []
    for horizon, week in enumerate(forecast_weeks, 1):
        week_rows = forecast_rows.loc[forecast_rows["semana"] == week].copy()
        week_features = preparar_features_semana(working_history, week_rows)
        features = week_features[FEATURES]
        feature_rows.append(features.copy())
        quantile_predictions = np.column_stack(
            [
                pipelines[quantile].predict(features)
                for quantile in PREDICTION_QUANTILES
            ]
        )
        quantile_predictions = np.sort(
            np.clip(quantile_predictions, 0, None), axis=1
        )

        result_columns = ["semana", "sku_id", "canal"]
        if TARGET in week_rows:
            result_columns.append(TARGET)
        if "preco_medio" in week_rows:
            result_columns.append("preco_medio")
        week_result = week_rows[result_columns].copy()
        week_result["previsao_p05"] = quantile_predictions[:, 0]
        week_result["previsao"] = quantile_predictions[:, 1]
        week_result["previsao_p95"] = quantile_predictions[:, 2]
        week_result["largura_faixa_pct"] = np.nan
        positive_predictions = week_result["previsao"] > 0
        week_result.loc[positive_predictions, "largura_faixa_pct"] = (
            (
                week_result.loc[positive_predictions, "previsao_p95"]
                - week_result.loc[positive_predictions, "previsao_p05"]
            )
            / week_result.loc[positive_predictions, "previsao"]
            * 100
        )
        week_result["horizonte"] = horizon
        results.append(week_result)

        predicted_history = week_rows[["sku_id", "canal", "semana"]].copy()
        predicted_history[TARGET] = week_result["previsao"].to_numpy()
        working_history = pd.concat(
            [working_history, predicted_history], ignore_index=True
        )

    forecast = pd.concat(results).sort_values(["semana", "sku_id", "canal"])
    forecast_features = pd.concat(feature_rows).loc[forecast.index]
    return forecast, forecast_features


def _evaluate_recursive_quantiles(
    model_name: str,
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    folds: list[tuple[np.ndarray, pd.Index]],
) -> pd.DataFrame:
    """Avalia os quantis em folds temporais após selecionar o modelo-base."""
    validation_predictions = []
    for train_indices, validation_weeks in folds:
        x_train = features.iloc[train_indices]
        y_train = target.iloc[train_indices]
        history = frame.iloc[train_indices]
        forecast_rows = frame.loc[frame["semana"].isin(validation_weeks)]
        if x_train.empty or forecast_rows.empty:
            raise ValueError("Fold temporal sem linhas de treino ou validação.")

        quantile_pipelines = {
            quantile: _make_pipeline(quantile_estimator)
            for quantile, quantile_estimator in _make_quantile_estimators(
                model_name, estimator
            ).items()
        }
        for pipeline in quantile_pipelines.values():
            pipeline.fit(x_train, y_train)
        predictions, _ = _forecast_recursively_quantiles(
            quantile_pipelines, history, forecast_rows
        )
        validation_predictions.append(predictions)
    return pd.concat(validation_predictions).sort_index()


def _quantile_metrics(
    actual: pd.Series, predictions: pd.DataFrame
) -> dict[str, object]:
    """Calcula pinball loss, cobertura por quantil e cobertura do intervalo."""
    quantile_columns = {
        0.05: "previsao_p05",
        0.50: "previsao",
        0.95: "previsao_p95",
    }
    return {
        "pinball_loss": {
            quantile: float(
                mean_pinball_loss(
                    actual,
                    predictions[column],
                    alpha=quantile,
                )
            )
            for quantile, column in quantile_columns.items()
        },
        "cobertura_por_quantil": {
            quantile: float(
                (actual.to_numpy() <= predictions[column].to_numpy()).mean()
            )
            for quantile, column in quantile_columns.items()
        },
        "cobertura_intervalo_p05_p95": float(
            (
                actual.ge(predictions["previsao_p05"])
                & actual.le(predictions["previsao_p95"])
            ).mean()
        ),
        "largura_media_intervalo": float(
            (
                predictions["previsao_p95"]
                - predictions["previsao_p05"]
            ).mean()
        ),
    }


def _quantile_metrics_by_horizon(
    predictions: pd.DataFrame,
) -> dict[int, dict[str, object]]:
    return {
        int(horizon): _quantile_metrics(group[TARGET], group)
        for horizon, group in predictions.groupby("horizonte", sort=True)
    }


def _evaluate_recursive_model(
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    folds: list[tuple[np.ndarray, pd.Index]],
) -> tuple[dict[str, float], dict[str, object], pd.DataFrame]:
    """Avalia cada fold prevendo recursivamente todo o horizonte de oito semanas."""
    train_targets = []
    train_predictions = []
    validation_predictions = []

    for train_indices, validation_weeks in folds:
        x_train = features.iloc[train_indices]
        y_train = target.iloc[train_indices]
        history = frame.iloc[train_indices]
        forecast_rows = frame.loc[
            frame["semana"].isin(validation_weeks)
        ]
        if x_train.empty or forecast_rows.empty:
            raise ValueError("Fold temporal sem linhas de treino ou validação.")

        pipeline = _make_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        train_prediction = np.clip(
            pipeline.predict(x_train), 0, None
        )
        train_targets.append(y_train)
        train_predictions.append(train_prediction)
        validation_prediction, _ = _forecast_recursively(
            pipeline, history, forecast_rows
        )
        validation_predictions.append(validation_prediction)

    oof_predictions = pd.concat(validation_predictions).sort_index()
    cross_validation_scores = _metrics(
        oof_predictions[TARGET],
        oof_predictions["previsao"].to_numpy(),
    )
    cross_validation_scores["por_horizonte"] = {
        int(horizon): _metrics(
            group[TARGET], group["previsao"].to_numpy()
        )
        for horizon, group in oof_predictions.groupby("horizonte", sort=True)
    }
    return (
        _metrics(
            pd.concat(train_targets),
            np.concatenate(train_predictions),
        ),
        cross_validation_scores,
        oof_predictions,
    )


def _random_search_best_model(
    model_name: str,
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    folds: list[tuple[np.ndarray, pd.Index]],
) -> tuple[
    BaseEstimator,
    dict[str, object],
    dict[str, float],
    dict[str, object],
    pd.DataFrame,
]:
    """Busca parâmetros avaliando cada opção no horizonte recursivo completo."""
    parameter_spaces = {
        "LightGBM": {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 5, 10],
            "min_child_samples": [10, 20, 40],
            "subsample": [0.7, 0.85, 1.0],
            "subsample_freq": [1, 5],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_lambda": [0, 1, 3],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.05, 0.1, 0.3],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [1, 3, 5],
        },
    }
    if model_name not in parameter_spaces:
        raise ValueError(f"Busca aleatória não configurada para {model_name}.")
    if model_name == "LightGBM":
        # Desativa somente as mensagens internas durante as muitas tentativas.
        estimator.set_params(verbosity=-1)

    sampled_parameters = ParameterSampler(
        parameter_spaces[model_name],
        n_iter=RANDOM_SEARCH_ITERATIONS,
        random_state=RANDOM_STATE,
    )
    best_model = None
    best_parameters = {}
    best_train_scores = {}
    best_cv_scores = {}
    best_predictions = pd.DataFrame()
    for parameters in sampled_parameters:
        candidate = clone(estimator).set_params(**parameters)
        train_scores, cv_scores, oof_predictions = _evaluate_recursive_model(
            candidate, frame, features, target, folds
        )
        candidate_score = (
            cv_scores["wape"],
            cv_scores["mae"],
        )
        best_score = (
            best_cv_scores.get("wape", float("inf")),
            best_cv_scores.get("mae", float("inf")),
        )
        if candidate_score < best_score:
            best_model = candidate
            best_parameters = parameters
            best_train_scores = train_scores
            best_cv_scores = cv_scores
            best_predictions = oof_predictions

    if best_model is None:
        raise ValueError(f"Nenhuma configuração avaliada para {model_name}.")
    return (
        best_model,
        best_parameters,
        best_train_scores,
        best_cv_scores,
        best_predictions,
    )


def run_model(df: pd.DataFrame) -> dict[str, object]:
    """Seleciona, ajusta e testa modelos em previsões recursivas de oito semanas."""
    missing_columns = sorted(set(FEATURES + [TARGET, "semana"]) - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Colunas necessárias ausentes nos dados preparados: "
            + ", ".join(missing_columns)
        )

    data = (
        df.dropna(subset=[TARGET, "semana"])
        .sort_values(["semana", "sku_id", "canal"])
        .reset_index(drop=True)
    )
    weeks = pd.Index(data["semana"].drop_duplicates().sort_values())
    required_weeks = MIN_TRAIN_WEEKS + CV_WEEKS + TEST_WEEKS
    if len(weeks) < required_weeks:
        raise ValueError(
            f"São necessárias ao menos {required_weeks} semanas "
            f"({MIN_TRAIN_WEEKS} de treino inicial, {CV_WEEKS} de validação "
            f"e {TEST_WEEKS} de teste); "
            f"foram encontradas {len(weeks)}."
        )

    test_weeks = weeks[-TEST_WEEKS:]
    cv_weeks = weeks[-CV_WEEKS - TEST_WEEKS : -TEST_WEEKS]
    pre_test_weeks = weeks[:-TEST_WEEKS]
    pre_test = data.loc[data["semana"].isin(pre_test_weeks)].reset_index(
        drop=True
    )
    test = data.loc[data["semana"].isin(test_weeks)].copy()
    folds = _make_temporal_folds(pre_test, cv_weeks)
    if len(folds) != CV_FOLDS:
        raise ValueError("Não foi possível formar os folds temporais esperados.")

    print(f"Alvo previsto: {TARGET}")
    print(
        f"Validação cruzada: {len(folds)} origens de previsão, "
        f"cada uma com horizonte recursivo de {FORECAST_HORIZON_WEEKS} "
        "semanas."
    )
    for fold_index, (train_indices, validation_period) in enumerate(folds, 1):
        fold_train = pre_test.iloc[train_indices]
        fold_validation = pre_test.loc[
            pre_test["semana"].isin(validation_period)
        ]
        print(
            f"Validação {fold_index}: treino até "
            f"{fold_train['semana'].max():%Y-%m-%d}; "
            f"horizonte {_period_description(fold_validation)}."
        )
    print(
        f"Teste final recursivo, sem atualizar com vendas reais: "
        f"{_period_description(test)}"
    )
    print(
        "Features futuras: calendário, SKU, canal, subcategoria e histórico "
        "de vendas previsto recursivamente; preço, promoção e ruptura futura "
        "não são usados."
    )

    features = pre_test[FEATURES]
    target = pre_test[TARGET]
    candidates: dict[str, BaseEstimator] = {
        "LightGBM": LGBMRegressor(
            verbosity=-1,
            random_state=RANDOM_STATE,
            importance_type="gain",
        ),
        "XGBoost": XGBRegressor(
            random_state=RANDOM_STATE,
            importance_type="total_gain",
        ),
    }
    model_results = {}
    recursive_evaluations = {}

    print("\nComparação dos modelos padrão nos folds temporais:")
    for name, estimator in candidates.items():
        train_scores, cv_scores, oof_predictions = _evaluate_recursive_model(
            estimator, pre_test, features, target, folds
        )
        recursive_evaluations[name] = (
            train_scores,
            cv_scores,
            oof_predictions,
        )
        model_results[name] = {
            "train": train_scores,
            "cross_validation": cv_scores,
        }
        print(
            f"{name:16s} | WAPE recursiva OOF em "
            f"{CV_WEEKS} semanas: {cv_scores['wape']:.2%} "
            f"| Erro médio (MAE): {cv_scores['mae']:.2f} unidades "
            f"| Tendência a super/subestimar (BIAS): {cv_scores['bias']:.2%}"
        )
        for horizon, horizon_scores in cv_scores["por_horizonte"].items():
            print(
                f"  Semana {horizon}: WAPE {horizon_scores['wape']:.2%}; "
                f"MAE {horizon_scores['mae']:.2f} un."
            )

    # Escolhe primeiro entre modelos padrão; o teste não participa da decisão.
    initial_winner = min(
        model_results,
        key=lambda name: (
            model_results[name]["cross_validation"]["wape"],
            model_results[name]["cross_validation"]["mae"],
        ),
    )
    print(f"\nMelhor modelo padrão na validação cruzada: {initial_winner}")

    print(
        f"\nBusca aleatória de {initial_winner}: "
        f"{RANDOM_SEARCH_ITERATIONS} combinações, avaliadas em previsões "
        "recursivas de oito semanas."
    )
    (
        tuned_estimator,
        best_parameters,
        tuned_train_scores,
        tuned_cv_scores,
        tuned_oof_predictions,
    ) = (
        _random_search_best_model(
            initial_winner,
            candidates[initial_winner],
            pre_test,
            features,
            target,
            folds,
        )
    )
    recursive_evaluations[f"{initial_winner} ajustado"] = (
        tuned_train_scores,
        tuned_cv_scores,
        tuned_oof_predictions,
    )
    print(
        "Menor WAPE recursiva encontrado na busca: "
        f"{tuned_cv_scores['wape']:.2%}"
    )
    print(
        f"WAPE recursiva OOF do {initial_winner} ajustado nas "
        f"{CV_WEEKS} semanas: {tuned_cv_scores['wape']:.2%}"
    )
    for horizon, horizon_scores in tuned_cv_scores["por_horizonte"].items():
        print(
            f"  Semana {horizon}: WAPE {horizon_scores['wape']:.2%}; "
            f"MAE {horizon_scores['mae']:.2f} un."
        )
    parameter_names = {
        "n_estimators": "quantidade de árvores",
        "max_depth": "profundidade máxima",
        "learning_rate": "taxa de aprendizado",
        "subsample": "proporção de linhas por árvore",
        "subsample_freq": "frequência de reamostragem de linhas",
        "colsample_bytree": "proporção de variáveis por árvore",
        "min_child_weight": "peso mínimo por folha",
        "reg_lambda": "regularização",
        "min_child_samples": "mínimo de linhas por folha",
        "num_leaves": "máximo de folhas",
    }
    readable_parameters = {
        parameter_names.get(key, key): value
        for key, value in best_parameters.items()
    }
    print(f"Melhor configuração da busca: {readable_parameters}")

    tuned_name = f"{initial_winner} ajustado"
    model_results[tuned_name] = {
        "train": tuned_train_scores,
        "cross_validation": tuned_cv_scores,
    }
    # Mantém o modelo padrão se a busca não superar o vencedor original.
    selected_name = min(
        [initial_winner, tuned_name],
        key=lambda name: (
            model_results[name]["cross_validation"]["wape"],
            model_results[name]["cross_validation"]["mae"],
        ),
    )
    selected_parameters = (
        best_parameters if selected_name == tuned_name else {}
    )
    print(f"\nModelo selecionado após o ajuste: {selected_name}")
    if selected_name == initial_winner:
        print("A configuração padrão superou a melhor configuração ajustada.")
    else:
        print(f"Configuração selecionada: {readable_parameters}")

    selected_estimator = (
        tuned_estimator
        if selected_name == tuned_name
        else candidates[initial_winner]
    )
    _, selected_cv_scores, _ = (
        recursive_evaluations[selected_name]
    )
    print(
        "\nAplicando regressão quantílica ao modelo-base já selecionado "
        f"({selected_name}); quantis: {PREDICTION_QUANTILES}."
    )
    quantile_oof_predictions = _evaluate_recursive_quantiles(
        initial_winner,
        selected_estimator,
        pre_test,
        features,
        target,
        folds,
    )
    quantile_cv_scores = _quantile_metrics(
        quantile_oof_predictions[TARGET], quantile_oof_predictions
    )
    quantile_cv_scores["por_horizonte"] = _quantile_metrics_by_horizon(
        quantile_oof_predictions
    )
    print(
        "Validação quantílica OOF | "
        f"pinball P05 {quantile_cv_scores['pinball_loss'][0.05]:.2f}; "
        f"P50 {quantile_cv_scores['pinball_loss'][0.50]:.2f}; "
        f"P95 {quantile_cv_scores['pinball_loss'][0.95]:.2f} | "
        f"cobertura P05–P95 "
        f"{quantile_cv_scores['cobertura_intervalo_p05_p95']:.1%} "
        f"(largura média "
        f"{quantile_cv_scores['largura_media_intervalo']:.2f} un.)"
    )

    # Ajusta novamente somente com dados anteriores ao teste; as semanas finais
    # não podem participar nem do ajuste nem da escolha dos parâmetros.
    quantile_pipelines = {
        quantile: _make_pipeline(quantile_estimator)
        for quantile, quantile_estimator in _make_quantile_estimators(
            initial_winner, selected_estimator
        ).items()
    }
    for pipeline in quantile_pipelines.values():
        pipeline.fit(pre_test[FEATURES], pre_test[TARGET])
    selected_pipeline = quantile_pipelines[0.50]
    train_prediction = np.clip(
        selected_pipeline.predict(pre_test[FEATURES]), 0, None
    )
    train_one_step_scores = _metrics(
        pre_test[TARGET], train_prediction
    )

    predictions, prediction_features = _forecast_recursively_quantiles(
        quantile_pipelines,
        pre_test[["sku_id", "canal", "semana", TARGET]],
        test,
    )

    test_scores = _metrics(
        test[TARGET],
        predictions["previsao"].to_numpy(),
    )
    test_scores["por_horizonte"] = {
        int(horizon): _metrics(
            group[TARGET], group["previsao"].to_numpy()
        )
        for horizon, group in predictions.groupby("horizonte", sort=True)
    }
    quantile_test_scores = _quantile_metrics(
        predictions[TARGET], predictions
    )
    quantile_test_scores["por_horizonte"] = _quantile_metrics_by_horizon(
        predictions
    )
    chart_paths = _save_test_forecast_charts(predictions)
    shap_explainer = shap.TreeExplainer(
        selected_pipeline.named_steps["model"]
    )
    shap_chart_paths = {}
    channels = {
        "Venda Direta": "venda_direta",
        "Loja": "loja",
    }
    for channel, filename in channels.items():
        examples = predictions.loc[
            predictions["sku_id"].eq("SKU_01")
            & predictions["canal"].eq(channel)
            & predictions["horizonte"].eq(1)
        ]
        if examples.empty:
            raise ValueError(
                f"Não há previsão de SKU_01 no horizonte 1 para o canal {channel}."
            )
        prediction_index = examples.index[0]
        shap_chart_paths[channel] = _save_shap_waterfall(
            selected_pipeline,
            shap_explainer,
            prediction_features.loc[[prediction_index]],
            examples.iloc[0],
            f"shap_waterfall_sku_01_{filename}.png",
        )

    print("\nResultado final do modelo selecionado:")
    print(f"Modelo: {selected_name}")
    print(f"Período usado para ajuste final: {_period_description(pre_test)}")
    print(
        "Referência de treino a uma semana (não recursiva) | "
        f"WAPE: {train_one_step_scores['wape']:.2%} "
        f"| MAE: {train_one_step_scores['mae']:.2f} unidades"
    )
    print(
        f"Teste recursivo | WAPE: {test_scores['wape']:.2%} "
        f"| Erro médio (MAE): {test_scores['mae']:.2f} unidades "
        f"| Tendência a super/subestimar (BIAS): {test_scores['bias']:.2%}"
    )
    print(
        "Cobertura observada do intervalo P05–P95 no teste: "
        f"{quantile_test_scores['cobertura_intervalo_p05_p95']:.1%} "
        f"| largura média: "
        f"{quantile_test_scores['largura_media_intervalo']:.2f} unidades"
    )
    print("Gráficos SHAP Waterfall (explicação de SKU_01 na semana 1 do teste):")
    for channel, chart_path in shap_chart_paths.items():
        print(f"  {channel}: {chart_path}")
    for horizon, horizon_scores in test_scores["por_horizonte"].items():
        print(
            f"  Semana {horizon}: WAPE {horizon_scores['wape']:.2%}; "
            f"MAE {horizon_scores['mae']:.2f} un."
        )
    print("Gráficos real vs. previsto por SKU-canal:")
    for channel, chart_path in chart_paths.items():
        print(f"  {channel}: {chart_path}")
    _print_sku_business_translation(predictions)
    display_columns = [
        "semana",
        "sku_id",
        "canal",
        "horizonte",
        TARGET,
        "previsao",
        "previsao_p05",
        "previsao_p95",
        "largura_faixa_pct",
    ]
    print("\nExemplos de previsões no teste:")
    print(predictions[display_columns].head(10).to_string(index=False))

    return {
        "model": selected_pipeline,
        "model_name": selected_name,
        "predictions": predictions,
        "metrics": {
            "train_one_step_reference": train_one_step_scores,
            "cross_validation": selected_cv_scores,
            "quantile_cross_validation": quantile_cv_scores,
            "test": test_scores,
            "quantile_test": quantile_test_scores,
            **test_scores,
        },
        "model_comparison": model_results,
        "best_hyperparameters": selected_parameters,
        "quantile_models": quantile_pipelines,
        "quantile_validation_predictions": quantile_oof_predictions,
        "shap_waterfall_charts": shap_chart_paths,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "periods": {
            "cross_validation": _period_description(
                pre_test.loc[pre_test["semana"].isin(cv_weeks)]
            ),
            "test": _period_description(test),
            "final_train": _period_description(pre_test),
        },
    }


def main() -> None:
    df = preparar_dados_modelo(carregar_dados_vendas())
    run_model(df)


if __name__ == "__main__":
    main()