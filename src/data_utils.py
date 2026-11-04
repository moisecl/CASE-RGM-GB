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


def load_sales_data() -> pd.DataFrame:
    """Carrega o CSV do projeto, independentemente do diretório de execução."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def prepare_model_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Limpa os dados e cria o alvo e as features temporais usados pelo modelo."""
    required = {
        "sku_id",
        "canal",
        "semana",
        *NUMERIC_COLUMNS,
        *FLAG_COLUMNS,
        *CATEGORY_COLUMNS,
    }
    missing = sorted(required.difference(raw_df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV: {', '.join(missing)}")

    df = raw_df.copy()
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

    df = df.drop_duplicates()
    aggregation = {column: "mean" for column in NUMERIC_COLUMNS}
    aggregation.update({column: "max" for column in FLAG_COLUMNS})
    aggregation.update({column: "first" for column in CATEGORY_COLUMNS})
    df = df.groupby(["sku_id", "canal", "semana"], as_index=False).agg(aggregation)

    df["qt_vendida_raw"] = df["qt_vendida"]
    df["qt_vendida"] = df["qt_vendida"].clip(lower=0)
    df["qt_ruptura"] = df["qt_ruptura"].fillna(0).clip(lower=0)

    median_price = df.groupby("sku_id")["preco_medio"].transform("median")
    df.loc[df["preco_medio"] > 5 * median_price, "preco_medio"] = np.nan
    df["preco_medio"] = df.groupby("sku_id")["preco_medio"].transform(
        lambda values: values.fillna(values.median())
    )

    df.loc[df["desconto_pct"] > 100, "desconto_pct"] = np.nan
    df["desconto_pct"] = df.groupby("sku_id")["desconto_pct"].transform(
        lambda values: values.fillna(values.median())
    )

    can_impute_quantity = (
        df["qt_vendida"].isna()
        & df["receita"].notna()
        & df["preco_medio"].gt(0)
    )
    df.loc[can_impute_quantity, "qt_vendida"] = (
        df.loc[can_impute_quantity, "receita"]
        / df.loc[can_impute_quantity, "preco_medio"]
    )

    df["subcategoria_id"] = df.groupby("sku_id")["subcategoria_id"].transform(
        lambda values: values.fillna(
            values.mode().iloc[0] if not values.mode().empty else "UNK"
        )
    )

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
    df["ruptura_flag_lag1"] = (previous_stockout > 0).astype(int)

    median_sku_price = df.groupby("sku_id")["preco_medio"].transform("median")
    df["preco_rel"] = df["preco_medio"] / median_sku_price
    df["wk_sin"] = np.sin(2 * np.pi * df["semana_iso"] / 52)
    df["wk_cos"] = np.cos(2 * np.pi * df["semana_iso"] / 52)
    df["dez"] = (df["mes"] == 12).astype(int)
    df["tendencia"] = (df["semana"] - df["semana"].min()).dt.days / 7.0
    return df
