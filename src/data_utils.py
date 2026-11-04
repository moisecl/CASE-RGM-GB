"""Carregamento e preparação comum dos dados do case."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "vendas_sku_semanal.csv"
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"

NUMERIC_COLUMNS = [
    "qt_vendida",
    "qt_ruptura",
    "receita",
    "preco_medio",
    "desconto_pct",
    "margem_pct",
]
FLAG_COLUMNS = [
    "flag_desconto_direto",
    "flag_combo",
    "flag_pague_e_leve",
]
CATEGORY_COLUMNS = ["categoria_id", "subcategoria_id"]
DATA_INICIO_MODELO = pd.Timestamp("2025-01-01")


def carregar_dados_vendas() -> pd.DataFrame:
    """Carrega o CSV de vendas a partir da pasta do projeto.

    Returns:
        DataFrame original, sem aplicar deduplicação ou tratamento de valores.

    Raises:
        FileNotFoundError: Se o CSV configurado não existir.
    """
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def remover_duplicatas_vendas(
    df: pd.DataFrame, colunas_chave: list[str]
) -> pd.DataFrame:
    """Remove linhas idênticas e seleciona um registro por chave de negócio.

    Para linhas diferentes com a mesma chave, prioriza maior completude;
    empates são resolvidos pela proximidade às medianas numéricas do grupo,
    preservando a ordem original quando ainda houver empate.

    Args:
        df: Dados de vendas a deduplicar.
        colunas_chave: Colunas que identificam unicamente uma observação.

    Returns:
        Cópia sem duplicatas exatas e com no máximo uma linha por chave.

    Raises:
        ValueError: Se alguma coluna-chave não estiver presente.
    """
    missing = sorted(set(colunas_chave).difference(df.columns))
    if missing:
        raise ValueError(f"Colunas-chave ausentes: {', '.join(missing)}")

    deduplicated = df.drop_duplicates()
    value_columns = [
        column for column in deduplicated.columns if column not in colunas_chave
    ]
    numeric_columns = [
        column
        for column in NUMERIC_COLUMNS
        if column in deduplicated.columns and column not in colunas_chave
    ]
    selected_positions = []

    for positions in deduplicated.groupby(
        colunas_chave, sort=False, dropna=False
    ).indices.values():
        group = deduplicated.iloc[positions]
        completeness = group[value_columns].notna().sum(axis=1).to_numpy()
        candidate_positions = np.flatnonzero(completeness == completeness.max())

        if len(candidate_positions) > 1 and numeric_columns:
            candidates = group.iloc[candidate_positions][numeric_columns]
            medians = group[numeric_columns].median()
            iqr = (
                group[numeric_columns].quantile(0.75)
                - group[numeric_columns].quantile(0.25)
            )
            scale = iqr.where(iqr.gt(0), medians.abs().clip(lower=1))
            distance = (
                candidates.sub(medians).abs().div(scale).sum(axis=1, min_count=1)
            ).fillna(0)
            candidate_positions = candidate_positions[
                int(np.argmin(distance.to_numpy()))
            ]
        else:
            candidate_positions = candidate_positions[0]

        selected_positions.append(positions[candidate_positions])

    return deduplicated.iloc[sorted(selected_positions)].copy()


def preparar_valores_ausentes_vendas(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as regras exploradas para corrigir valores e imputar ausências.

    Converte quantidade negativa em valor absoluto para qt_vendida, recalcula
    preços acima de cinco vezes a mediana do SKU usando receita / quantidade
    quando o preço calculado é positivo e permanece dentro desse limite,
    recalcula quantidade/receita/preço quando um dos três campos está ausente, e preenche
    subcategoria apenas quando o mapeamento SKU-categoria-canal é único.
    Imputa margem e desconto pela mediana SKU-canal entre registros com receita
    positiva; descontos acima de 100% são tratados como inválidos. Preço ainda
    ausente após as recuperações recebe a mediana do SKU. Valores negativos de
    ruptura são limitados a zero, mas ausências de ruptura são preservadas.

    Args:
        df: Dados já deduplicados e com a data normalizada.

    Returns:
        Cópia dos dados com as transformações aplicadas.
    """
    df = df.copy()
    if "qt_vendida_raw" not in df.columns:
        df["qt_vendida_raw"] = df["qt_vendida"]
    df["qt_vendida"] = df["qt_vendida"].abs()
    df["qt_ruptura"] = df["qt_ruptura"].clip(lower=0)

    median_price = df.groupby("sku_id")["preco_medio"].transform("median")
    price_upper_limit = 5 * median_price
    price_outlier = df["preco_medio"] > price_upper_limit
    df.loc[price_outlier, "preco_medio"] = np.nan

    calculated_price = (
        df["receita"] / df["qt_vendida"].replace(0, np.nan)
    )
    can_correct_price = (
        df["preco_medio"].isna()
        & df["receita"].notna()
        & df["qt_vendida"].gt(0)
        & calculated_price.gt(0)
        & calculated_price.le(price_upper_limit)
    )
    df.loc[can_correct_price, "preco_medio"] = calculated_price.loc[
        can_correct_price
    ]

    can_impute_quantity = (
        df["qt_vendida"].isna()
        & df["receita"].notna()
        & df["preco_medio"].gt(0)
    )
    df.loc[can_impute_quantity, "qt_vendida"] = (
        df.loc[can_impute_quantity, "receita"]
        / df.loc[can_impute_quantity, "preco_medio"]
    )

    can_impute_revenue = (
        df["receita"].isna()
        & df["qt_vendida"].notna()
        & df["preco_medio"].notna()
    )
    df.loc[can_impute_revenue, "receita"] = (
        df.loc[can_impute_revenue, "qt_vendida"]
        * df.loc[can_impute_revenue, "preco_medio"]
    )

    subcategory_keys = ["sku_id", "categoria_id", "canal"]
    subcategory_groups = df.groupby(subcategory_keys, dropna=False)[
        "subcategoria_id"
    ]
    subcategory_count = subcategory_groups.transform("nunique")
    subcategory_value = subcategory_groups.transform("first")
    can_impute_subcategory = (
        df["subcategoria_id"].isna() & subcategory_count.eq(1)
    )
    df.loc[can_impute_subcategory, "subcategoria_id"] = subcategory_value[
        can_impute_subcategory
    ]

    df["preco_medio"] = df.groupby("sku_id")["preco_medio"].transform(
        lambda values: values.fillna(values.median())
    )

    df.loc[df["desconto_pct"] > 100, "desconto_pct"] = np.nan
    revenue_positive = df["receita"].gt(0)
    grouping_columns = [df["sku_id"], df["canal"]]
    for column in ["margem_pct", "desconto_pct"]:
        group_median = (
            df[column]
            .where(revenue_positive)
            .groupby(grouping_columns, dropna=False)
            .transform("median")
        )
        can_impute = df[column].isna() & revenue_positive
        df.loc[can_impute, column] = group_median.loc[can_impute]

    return df


def preparar_dados_modelo(dados_brutos: pd.DataFrame) -> pd.DataFrame:
    """Valida, limpa e cria alvo e atributos temporais para a modelagem.

    Valida as colunas obrigatórias e as datas, aplica deduplicação e as regras
    de tratamento de valores de ``preparar_valores_ausentes_vendas``, calcula
    demanda real, calendário e atributos históricos (lags e janelas móveis).
    Descarta observações anteriores a 2025 antes de criar atributos temporais,
    pois 2024 tem cobertura parcial e comportamento não comparável à série
    observada de 2025–2026.

    Args:
        dados_brutos: Dados de vendas carregados do CSV.

    Returns:
        DataFrame preparado para treino, validação e inferência.

    Raises:
        ValueError: Se faltarem colunas obrigatórias ou houver datas inválidas.
    """
    required = {
        "sku_id",
        "canal",
        "semana",
        *NUMERIC_COLUMNS,
        *FLAG_COLUMNS,
        *CATEGORY_COLUMNS,
    }
    missing = sorted(required.difference(dados_brutos.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV: {', '.join(missing)}")

    df = dados_brutos.copy()
    dates = df["semana"].astype("string")
    iso_dates = pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce")
    br_dates = pd.to_datetime(dates, format="%d/%m/%Y", errors="coerce")
    df["semana"] = iso_dates.fillna(br_dates)
    invalid_dates = df["semana"].isna()
    if invalid_dates.any():
        rows = (invalid_dates[invalid_dates].index + 2).tolist()[:5]
        raise ValueError(
            "Há datas inválidas na coluna 'semana'; "
            f"linhas do CSV (amostra): {rows}"
        )

    df = df.loc[df["semana"] >= DATA_INICIO_MODELO].copy()
    if df.empty:
        raise ValueError(
            "Não há observações a partir de "
            f"{DATA_INICIO_MODELO.date()} para preparar o modelo."
        )

    df = remover_duplicatas_vendas(df, ["sku_id", "canal", "semana"])
    df = preparar_valores_ausentes_vendas(df)

    df["demanda_real"] = df["qt_vendida"] + df["qt_ruptura"]
    median_demand = df.groupby("sku_id")["demanda_real"].transform("median")
    df.loc[
        (df["demanda_real"] == 0) & (median_demand > 20), "demanda_real"
    ] = np.nan

    iso_calendar = df["semana"].dt.isocalendar()
    df["ano"] = iso_calendar.year.astype(int)
    df["semana_iso"] = iso_calendar.week.astype(int)
    df["mes"] = df["semana"].dt.month

    df = df.sort_values(["sku_id", "canal", "semana"]).copy()

    series_groups = df.groupby(["sku_id", "canal"], sort=False)
    demand = series_groups["demanda_real"]
    for lag in [1, 2, 4, 8]:
        df[f"lag_{lag}"] = demand.shift(lag)
    df["roll_mean_4"] = demand.transform(
        lambda values: values.shift(1).rolling(4).mean()
    )
    df["roll_mean_8"] = demand.transform(
        lambda values: values.shift(1).rolling(8).mean()
    )
    df["roll_std_4"] = demand.transform(
        lambda values: values.shift(1).rolling(4).std()
    )
    previous_stockout = series_groups["qt_ruptura"].shift(1)
    df["ruptura_lag1"] = previous_stockout
    df["ruptura_flag_lag1"] = np.where(
        previous_stockout.isna(),
        np.nan,
        previous_stockout.gt(0).astype(int),
    )

    median_sku_price = df.groupby("sku_id")["preco_medio"].transform("median")
    df["preco_rel"] = df["preco_medio"] / median_sku_price
    df["wk_sin"] = np.sin(2 * np.pi * df["semana_iso"] / 52)
    df["wk_cos"] = np.cos(2 * np.pi * df["semana_iso"] / 52)
    df["dez"] = (df["mes"] == 12).astype(int)
    df["tendencia"] = (df["semana"] - df["semana"].min()).dt.days / 7.0
    return df
