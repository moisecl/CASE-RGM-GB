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
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
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
PREDICTION_BUCKETS = ("<50", "50-150", ">150")
ERROR_QUANTILE = 0.90
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
                group["previsao_inferior"],
                group["previsao_superior"],
                alpha=0.15,
                label="Faixa estimada (P90)",
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
) -> pd.DataFrame:
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
    for horizon, week in enumerate(forecast_weeks, 1):
        week_rows = forecast_rows.loc[forecast_rows["semana"] == week].copy()
        week_features = preparar_features_semana(working_history, week_rows)
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

    return pd.concat(results).sort_values(["semana", "sku_id", "canal"])


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
        validation_predictions.append(
            _forecast_recursively(pipeline, history, forecast_rows)
        )

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


def _prediction_buckets(predictions: pd.Series) -> pd.Series:
    """Classifica previsões nas faixas de volume usadas na calibração."""
    values = predictions.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("As previsões devem ser finitas para calcular a faixa.")

    return pd.Series(
        np.select(
            [values < 50, values <= 150],
            [PREDICTION_BUCKETS[0], PREDICTION_BUCKETS[1]],
            default=PREDICTION_BUCKETS[2],
        ),
        index=predictions.index,
        name="faixa_previsao",
    )


def _oof_error_by_bucket(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    """Calcula o erro P90 por semana do horizonte e porte da previsão."""
    buckets = _prediction_buckets(oof_predictions["previsao"])
    absolute_errors = (
        oof_predictions[TARGET] - oof_predictions["previsao"]
    ).abs()
    summary = []
    for horizon in range(1, FORECAST_HORIZON_WEEKS + 1):
        for bucket in PREDICTION_BUCKETS:
            mask = oof_predictions["horizonte"].eq(horizon) & buckets.eq(bucket)
            errors = absolute_errors.loc[mask]
            summary.append(
                {
                    "horizonte": horizon,
                    "faixa_previsao": bucket,
                    "quantidade_oof": len(errors),
                    "erro_p90_unidades": (
                        float(errors.quantile(ERROR_QUANTILE))
                        if not errors.empty
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(summary).set_index(["horizonte", "faixa_previsao"])


def _add_prediction_intervals(
    predictions: pd.DataFrame, error_by_bucket: pd.DataFrame
) -> pd.DataFrame:
    """Anexa faixa P90 calibrada por horizonte e porte da previsão."""
    result = predictions.copy()
    prediction_buckets = _prediction_buckets(result["previsao"])
    calibration_keys = pd.MultiIndex.from_arrays(
        [result["horizonte"], prediction_buckets],
        names=["horizonte", "faixa_previsao"],
    )
    result["erro_p90_unidades"] = error_by_bucket[
        "erro_p90_unidades"
    ].reindex(calibration_keys).to_numpy()
    result["previsao_inferior"] = (
        result["previsao"] - result["erro_p90_unidades"]
    ).clip(lower=0)
    result["previsao_superior"] = (
        result["previsao"] + result["erro_p90_unidades"]
    )
    result["largura_faixa_pct"] = np.nan
    positive_predictions = result["previsao"] > 0
    result.loc[positive_predictions, "largura_faixa_pct"] = (
        2
        * result.loc[positive_predictions, "erro_p90_unidades"]
        / result.loc[positive_predictions, "previsao"]
        * 100
    )
    return result


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
        "LightGBM": LGBMRegressor(verbosity=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE),
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
    selected_pipeline = _make_pipeline(clone(selected_estimator))
    _, selected_cv_scores, oof_predictions = (
        recursive_evaluations[selected_name]
    )
    error_by_bucket = _oof_error_by_bucket(oof_predictions)
    print(
        f"\nErro absoluto P{ERROR_QUANTILE:.0%} por semana do horizonte e "
        f"faixa de previsão (unidades):\n"
        + error_by_bucket.to_string(
            formatters={
                "erro_p90_unidades": lambda value: f"{value:.2f}"
            }
        )
    )

    # Ajusta novamente somente com dados anteriores ao teste; as semanas finais
    # não podem participar nem do ajuste nem da escolha dos parâmetros.
    selected_pipeline.fit(pre_test[FEATURES], pre_test[TARGET])
    train_prediction = np.clip(
        selected_pipeline.predict(pre_test[FEATURES]), 0, None
    )
    train_one_step_scores = _metrics(
        pre_test[TARGET], train_prediction
    )

    predictions = _forecast_recursively(
        selected_pipeline,
        pre_test[["sku_id", "canal", "semana", TARGET]],
        test,
    )
    predictions = _add_prediction_intervals(predictions, error_by_bucket)

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
    chart_paths = _save_test_forecast_charts(predictions)

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
        "erro_p90_unidades",
        "previsao_inferior",
        "previsao_superior",
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
            "test": test_scores,
            **test_scores,
        },
        "model_comparison": model_results,
        "best_hyperparameters": selected_parameters,
        "erro_oof_por_horizonte_e_faixa": error_by_bucket,
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