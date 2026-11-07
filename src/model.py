"""Compara modelos de venda semanal por validação cruzada temporal.

Modelos:
* LightGBM: rápido e eficiente em dados tabulares; modelos muito complexos
  podem sobreajustar e árvores não extrapolam tendências.
* XGBoost: boosting flexível, com regularização e bom desempenho tabular;
  pode exigir mais tempo/recursos e árvores também não extrapolam tendências.

Referências:
* LightGBM: https://lightgbm.readthedocs.io/en/stable/
* XGBoost: https://xgboost.readthedocs.io/en/stable/
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

if __package__:
    from src.data_utils import carregar_dados_vendas, preparar_dados_modelo
else:
    from src.data_utils import carregar_dados_vendas, preparar_dados_modelo


TARGET = "qt_vendida"
CV_WEEKS = 16
FOLD_WEEKS = 4
TEST_WEEKS = 8
RANDOM_STATE = 42
INTERVAL_COVERAGE = 0.90
RANDOM_SEARCH_ITERATIONS = 12

# Preço e mecânicas promocionais precisam ser conhecidos no momento da previsão.
NUMERIC_FEATURES = [
    "preco_medio",
    "desconto_pct",
    "flag_desconto_direto",
    "flag_combo",
    "flag_pague_e_leve",
    "wk_sin",
    "wk_cos",
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
]
CATEGORICAL_FEATURES = ["sku_id", "canal", "subcategoria_id"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Dicionário para tornar a entrada do modelo compreensível e auditável.
FEATURE_DESCRIPTIONS = {
    "preco_medio": "Preço médio da semana; deve ser conhecido antes da previsão.",
    "desconto_pct": "Percentual de desconto planejado para a semana.",
    "flag_desconto_direto": "Indica promoção de desconto direto.",
    "flag_combo": "Indica participação em promoção do tipo combo.",
    "flag_pague_e_leve": "Indica promoção do tipo pague e leve.",
    "wk_sin": "Componente seno que representa a posição cíclica da semana no ano.",
    "wk_cos": "Componente cosseno que representa a posição cíclica da semana no ano.",
    "tendencia": "Índice de tempo em semanas desde o início da série preparada.",
    "lag_1": "Quantidade vendida na semana anterior para o mesmo SKU e canal.",
    "lag_2": "Quantidade vendida duas semanas antes para o mesmo SKU e canal.",
    "lag_4": "Quantidade vendida quatro semanas antes para o mesmo SKU e canal.",
    "lag_8": "Quantidade vendida oito semanas antes para o mesmo SKU e canal.",
    "roll_mean_4": "Média das quatro quantidades anteriores, sem incluir a semana prevista.",
    "roll_mean_8": "Média das oito quantidades anteriores, sem incluir a semana prevista.",
    "roll_std_4": "Variação das quatro quantidades anteriores.",
    "ruptura_lag1": "Quantidade em ruptura na semana anterior.",
    "ruptura_flag_lag1": "Indica se houve ruptura na semana anterior.",
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


def _period_description(frame: pd.DataFrame) -> str:
    """Descreve as datas cobertas, semanas e linhas de um período."""
    return (
        f"{frame['semana'].min():%Y-%m-%d} a "
        f"{frame['semana'].max():%Y-%m-%d} "
        f"({frame['semana'].nunique()} semanas, {len(frame)} linhas)"
    )


def _make_temporal_folds(
    frame: pd.DataFrame, weeks: pd.Index
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cria folds expansivos contíguos que cobrem todas as semanas de CV."""
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
        validation_indices = np.flatnonzero(
            frame["semana"].isin(validation_dates)
        )
        if not len(train_indices):
            raise ValueError("O fold temporal não contém semanas de treino.")
        if train_dates.max() >= validation_dates.min():
            raise ValueError("O fold temporal inclui treino posterior à validação.")
        if len(validation_dates) != FOLD_WEEKS:
            raise ValueError(
                f"O fold não contém exatamente {FOLD_WEEKS} semanas."
            )
        folds.append((train_indices, validation_indices))
    return folds


def _negative_wape(
    estimator: Pipeline, features: pd.DataFrame, actual: pd.Series
) -> float:
    """Pontua a busca pelo negativo da WAPE em todas as linhas."""
    predicted = np.clip(estimator.predict(features), 0, None)
    return -_metrics(actual, predicted)["wape"]


def _evaluate_model(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], dict[str, float], pd.Series]:
    """Avalia um modelo nos folds e devolve métricas e previsões fora da amostra."""
    out_of_fold = pd.Series(np.nan, index=features.index, dtype=float)
    train_wapes = []

    for train_indices, validation_indices in folds:
        x_train = features.iloc[train_indices]
        y_train = target.iloc[train_indices]
        x_validation = features.iloc[validation_indices]
        if x_train.empty or x_validation.empty:
            raise ValueError("Fold temporal sem linhas de treino ou validação.")

        pipeline = _make_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        train_prediction = np.clip(
            pipeline.predict(x_train), 0, None
        )
        validation_prediction = np.clip(
            pipeline.predict(x_validation), 0, None
        )
        train_wapes.append(
            _metrics(y_train, train_prediction)["wape"]
        )
        out_of_fold.loc[x_validation.index] = validation_prediction

    scored = out_of_fold.notna()
    scores = _metrics(target.loc[scored], out_of_fold.loc[scored].to_numpy())
    return (
        {"wape": float(np.mean(train_wapes))},
        scores,
        out_of_fold,
    )


def _interval_radius(errors: np.ndarray, coverage: float) -> float:
    """Calcula a largura do intervalo pelo quantil dos erros dos folds."""
    sorted_errors = np.sort(errors)
    if not len(sorted_errors):
        raise ValueError("Não há erros dos folds para estimar a confiabilidade.")

    rank = int(np.ceil((len(sorted_errors) + 1) * coverage))
    return (
        float("inf")
        if rank > len(sorted_errors)
        else float(sorted_errors[rank - 1])
    )


def _random_search_best_model(
    model_name: str,
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[BaseEstimator, dict[str, object], float]:
    """Ajusta por busca aleatória apenas o modelo que venceu a validação."""
    parameter_spaces = {
        "LightGBM": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__num_leaves": [15, 31, 63],
            "model__max_depth": [-1, 5, 10],
            "model__min_child_samples": [10, 20, 40],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__subsample_freq": [1, 5],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [0, 1, 3],
        },
        "XGBoost": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [3, 4, 6],
            "model__learning_rate": [0.03, 0.05, 0.1, 0.3],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_weight": [1, 3, 5],
            "model__reg_lambda": [1, 3, 5],
        },
    }
    if model_name not in parameter_spaces:
        raise ValueError(f"Busca aleatória não configurada para {model_name}.")
    if model_name == "LightGBM":
        # Desativa somente as mensagens internas durante as muitas tentativas.
        estimator.set_params(verbosity=-1)

    search = RandomizedSearchCV(
        estimator=_make_pipeline(estimator),
        param_distributions=parameter_spaces[model_name],
        n_iter=RANDOM_SEARCH_ITERATIONS,
        scoring=_negative_wape,
        cv=folds,
        refit=False,
        random_state=RANDOM_STATE,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(features, target)
    best_model = clone(estimator)
    model_parameters = {
        key.removeprefix("model__"): value
        for key, value in search.best_params_.items()
    }
    best_model.set_params(**model_parameters)
    return best_model, search.best_params_, float(-search.best_score_)


def run_model(df: pd.DataFrame) -> dict[str, object]:
    """Seleciona, ajusta e testa modelos com validação cruzada temporal."""
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
    required_weeks = CV_WEEKS + TEST_WEEKS + 1
    if len(weeks) < required_weeks:
        raise ValueError(
            f"São necessárias ao menos {required_weeks} semanas; "
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
    expected_folds = CV_WEEKS // FOLD_WEEKS
    if len(folds) != expected_folds:
        raise ValueError("Não foi possível formar os folds temporais esperados.")

    print(f"Alvo previsto: {TARGET}")
    print(
        f"Validação cruzada: janela expansiva de {CV_WEEKS} semanas, "
        f"com {len(folds)} blocos contíguos de {FOLD_WEEKS} semanas."
    )
    for fold_index, (train_indices, validation_indices) in enumerate(folds, 1):
        fold_train = pre_test.iloc[train_indices]
        fold_validation = pre_test.iloc[validation_indices]
        print(
            f"Fold {fold_index}: treino {_period_description(fold_train)}; "
            f"validação {_period_description(fold_validation)}."
        )
    print(f"Teste final intocado: {_period_description(test)}")
    print("NaN numéricos são tratados nativamente pelos modelos de boosting.")
    print(
        "Os folds também estimam a faixa provável da previsão; "
        "não é necessário reservar outro período."
    )

    features = pre_test[FEATURES]
    target = pre_test[TARGET]
    candidates: dict[str, BaseEstimator] = {
        "LightGBM": LGBMRegressor(verbosity=-1),
        "XGBoost": XGBRegressor(),
    }
    model_results = {}
    out_of_fold_predictions = {}

    print("\nComparação dos modelos padrão nos folds temporais:")
    for name, estimator in candidates.items():
        train_scores, cv_scores, oof_predictions = _evaluate_model(
            estimator, features, target, folds
        )
        model_results[name] = {
            "train": train_scores,
            "cross_validation": cv_scores,
        }
        out_of_fold_predictions[name] = oof_predictions
        print(
            f"{name:16s} | erro treino: {train_scores['wape']:.2%} "
            f"| WAPE OOF nas {CV_WEEKS} semanas: {cv_scores['wape']:.2%} "
            f"| erro médio: {cv_scores['mae']:.2f} unidades "
            f"| tendência a super/subestimar: {cv_scores['bias']:.2%}"
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
        f"{RANDOM_SEARCH_ITERATIONS} combinações nos folds temporais."
    )
    tuned_estimator, best_parameters, search_cv_wape = (
        _random_search_best_model(
            initial_winner,
            candidates[initial_winner],
            features,
            target,
            folds,
        )
    )
    tuned_train_scores, tuned_cv_scores, tuned_oof = _evaluate_model(
        tuned_estimator, features, target, folds
    )
    print(f"Menor erro médio encontrado na busca: {search_cv_wape:.2%}")
    print(
        f"WAPE OOF de {initial_winner} ajustado nas {CV_WEEKS} semanas: "
        f"{tuned_cv_scores['wape']:.2%}"
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
        parameter_names.get(key.removeprefix("model__"), key): value
        for key, value in best_parameters.items()
    }
    print(f"Configuração escolhida (nomes simplificados): {readable_parameters}")

    tuned_name = f"{initial_winner} ajustado"
    model_results[tuned_name] = {
        "train": tuned_train_scores,
        "cross_validation": tuned_cv_scores,
    }
    out_of_fold_predictions[tuned_name] = tuned_oof
    # Mantém o modelo padrão se a busca não superar o vencedor original.
    selected_name = min(
        [initial_winner, tuned_name],
        key=lambda name: (
            model_results[name]["cross_validation"]["wape"],
            model_results[name]["cross_validation"]["mae"],
        ),
    )
    print(f"\nModelo selecionado após o ajuste: {selected_name}")

    if selected_name == tuned_name:
        selected_pipeline = _make_pipeline(tuned_estimator)
    else:
        selected_pipeline = _make_pipeline(candidates[initial_winner])

    selected_oof = out_of_fold_predictions[selected_name]
    oof_valid = selected_oof.notna()
    oof_errors = np.abs(
        target.loc[oof_valid].to_numpy()
        - selected_oof.loc[oof_valid].to_numpy()
    )
    interval_radius = _interval_radius(oof_errors, INTERVAL_COVERAGE)

    # Ajusta novamente somente com dados anteriores ao teste; as semanas finais
    # não podem participar nem do ajuste nem da escolha dos parâmetros.
    selected_pipeline.fit(pre_test[FEATURES], pre_test[TARGET])
    train_prediction = np.clip(
        selected_pipeline.predict(pre_test[FEATURES]), 0, None
    )
    train_scores = _metrics(pre_test[TARGET], train_prediction)

    predictions = test.copy()
    predictions["pred"] = np.clip(
        selected_pipeline.predict(test[FEATURES]), 0, None
    )
    predictions["pred_lo"] = np.maximum(
        predictions["pred"] - interval_radius, 0
    )
    predictions["pred_hi"] = predictions["pred"] + interval_radius
    interval_width = predictions["pred_hi"] - predictions["pred_lo"]
    relative_width = interval_width / predictions["pred"].clip(lower=1)
    predictions["largura_relativa_intervalo"] = relative_width
    predictions["confiabilidade"] = np.select(
        [relative_width <= 0.5, relative_width <= 1.0],
        ["Alta", "Média"],
        default="Baixa",
    )

    test_scores = _metrics(
        test[TARGET],
        predictions["pred"].to_numpy(),
    )
    test_coverage = float(
        (
            (test[TARGET] >= predictions["pred_lo"])
            & (test[TARGET] <= predictions["pred_hi"])
        ).mean()
    )

    print("\nResultado final do modelo selecionado:")
    print(f"Modelo: {selected_name}")
    print(f"Período usado para ajuste final: {_period_description(pre_test)}")
    print(
        f"Treino | erro percentual: {train_scores['wape']:.2%} "
        f"| erro médio: {train_scores['mae']:.2f} unidades "
        f"| tendência a super/subestimar: {train_scores['bias']:.2%}"
    )
    print(
        f"Teste  | erro percentual: {test_scores['wape']:.2%} "
        f"| erro médio: {test_scores['mae']:.2f} unidades "
        f"| tendência a super/subestimar: {test_scores['bias']:.2%}"
    )
    print(
        f"Faixa estimada para {INTERVAL_COVERAGE:.0%} das previsões: "
        f"até +/- {interval_radius:.2f} unidades, calculada com os erros "
        f"fora da amostra das {CV_WEEKS} semanas de validação."
    )
    print(
        f"Na prática, a faixa cobriu {test_coverage:.1%} das vendas reais "
        "no teste final."
    )
    print(
        "Confiabilidade: Alta quando a faixa tem até 50% da previsão; "
        "Média até 100%; Baixa acima de 100%. A categoria indica a largura "
        "relativa da faixa, não uma probabilidade individual de acerto."
    )
    print(
        "Distribuição de confiabilidade: "
        f"{predictions['confiabilidade'].value_counts().to_dict()}"
    )
    display_columns = [
        "semana",
        "sku_id",
        "canal",
        TARGET,
        "pred",
        "pred_lo",
        "pred_hi",
        "confiabilidade",
    ]
    print("\nExemplos de previsão, faixa plausível e grau de confiabilidade:")
    print(predictions[display_columns].head(10).to_string(index=False))

    return {
        "model": selected_pipeline,
        "model_name": selected_name,
        "predictions": predictions,
        "metrics": {
            "train": train_scores,
            "cross_validation": model_results[selected_name]["cross_validation"],
            "test": test_scores,
            "test_coverage": test_coverage,
            "interval_radius": interval_radius,
            **test_scores,
        },
        "model_comparison": model_results,
        "best_hyperparameters": best_parameters,
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