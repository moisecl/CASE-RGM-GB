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


def _tratar_outlier_preco(
    df: pd.DataFrame, causal_imputation: bool
) -> pd.DataFrame:
    """Detecta preço extremo e, na EDA, tenta reconstruí-lo por receita/quantidade."""
    if causal_imputation:
        price_baseline = df.groupby(
            ["sku_id", "canal"], sort=False
        )["preco_medio"].transform(
            lambda values: values.shift(1).expanding().median()
        )
        # Na primeira semana da série, a mediana anterior é NaN; comparar com
        # o limite NaN retorna falso, então o preço inicial não é classificado.
    else:
        price_baseline = df.groupby("sku_id")["preco_medio"].transform("median")

    price_upper_limit = 5 * price_baseline
    price_outlier = df["preco_medio"] > price_upper_limit
    df.loc[price_outlier, "preco_medio"] = np.nan

    if not causal_imputation:
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
    return df


def _imputar_qtd_receita_preco(
    df: pd.DataFrame, causal_imputation: bool
) -> pd.DataFrame:
    """Imputa preço e recupera quantidade/receita antes de descartar casos inválidos.

    No modo causal, usa somente preços históricos; se não houver referência
    anterior, preserva o NaN para o tratamento nativo do LightGBM/XGBoost.
    """
    if "qt_vendida_raw" not in df.columns:
        df["qt_vendida_raw"] = df["qt_vendida"]
    df["qt_vendida"] = df["qt_vendida"].abs()

    no_sales_missing_revenue = (
        df["qt_vendida"].eq(0) & df["receita"].isna()
    )
    df.loc[no_sales_missing_revenue, "receita"] = 0
    df = _tratar_outlier_preco(df, causal_imputation)

    # A EDA aceita estatísticas da série completa. O modelo usa apenas preços
    # anteriores e mantém NaN quando não há histórico para fazer a imputação.
    if causal_imputation:
        historical_price = df.groupby(
            ["sku_id", "canal"], sort=False
        )["preco_medio"].transform(
            lambda values: values.shift(1).expanding().median()
        )
        df["preco_medio"] = df["preco_medio"].fillna(historical_price)
    else:
        sku_price_median = df.groupby("sku_id")["preco_medio"].transform(
            "median"
        )
        df["preco_medio"] = df["preco_medio"].fillna(sku_price_median)

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

    # Recupera quantidade/receita quando possível; remove apenas os registros
    # que continuam sem target ou sem receita depois das tentativas.
    df = df.dropna(subset=["qt_vendida", "receita"]).copy()
    remaining_missing_prices = int(df["preco_medio"].isna().sum())
    if remaining_missing_prices and not causal_imputation:
        missing_skus = (
            df.loc[df["preco_medio"].isna(), "sku_id"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"{remaining_missing_prices} preços continuam ausentes na EDA "
            "após a mediana do SKU; SKUs afetados: "
            f"{missing_skus[:10]}."
        )
    return df


def _imputar_subcategoria(
    df: pd.DataFrame, causal_imputation: bool
) -> pd.DataFrame:
    """Preenche subcategoria por mapeamento único ou já observado no passado."""
    grouping_columns = ["sku_id", "categoria_id", "canal"]
    subcategory_groups = df.groupby(
        grouping_columns, dropna=False, sort=False
    )["subcategoria_id"]
    if causal_imputation:
        # No modelo, só pode ser propagado um mapeamento disponível até a data.
        prior_subcategory = subcategory_groups.transform("ffill")
        can_impute = df["subcategoria_id"].isna() & prior_subcategory.notna()
        df.loc[can_impute, "subcategoria_id"] = prior_subcategory.loc[can_impute]
    else:
        # Na EDA, o mapeamento único pode ser confirmado usando todo o histórico.
        unique_mapping = subcategory_groups.transform("nunique").eq(1)
        first_value = subcategory_groups.transform("first")
        can_impute = df["subcategoria_id"].isna() & unique_mapping
        df.loc[can_impute, "subcategoria_id"] = first_value.loc[can_impute]
    return df


def _imputar_margem_desconto(
    df: pd.DataFrame, causal_imputation: bool
) -> pd.DataFrame:
    """Trata desconto inválido e preenche margem/desconto segundo o modo escolhido."""
    df.loc[df["desconto_pct"] > 100, "desconto_pct"] = np.nan
    revenue_positive = df["receita"].gt(0)
    grouping_columns = [df["sku_id"], df["canal"]]

    for column in ["margem_pct", "desconto_pct"]:
        valid_values = df[column].where(revenue_positive)
        if causal_imputation:
            group_median = valid_values.groupby(
                grouping_columns, dropna=False, sort=False
            ).transform(lambda values: values.shift(1).expanding().median())
            can_impute = df[column].isna() & group_median.notna()
        else:
            group_median = valid_values.groupby(
                grouping_columns, dropna=False
            ).transform("median")
            can_impute = df[column].isna() & revenue_positive
        df.loc[can_impute, column] = group_median.loc[can_impute]
    return df


def preparar_valores_ausentes_vendas(
    df: pd.DataFrame, causal_imputation: bool = False
) -> pd.DataFrame:
    """Aplica limpeza e imputação de ausências em etapas auditáveis.

    A EDA pode usar a distribuição completa para imputar estatísticas. A
    preparação do modelo ativa o modo causal, que usa somente observações
    anteriores para imputar features e preserva ausências de ruptura.
    """
    df = df.copy()
    if causal_imputation:
        df = df.sort_values(["sku_id", "canal", "semana"]).copy()

    df = _imputar_qtd_receita_preco(df, causal_imputation)
    df = _imputar_subcategoria(df, causal_imputation)
    return _imputar_margem_desconto(df, causal_imputation)


def preparar_dados_modelo(dados_brutos: pd.DataFrame) -> pd.DataFrame:
    """Valida, limpa e cria alvo e atributos temporais para a modelagem.

    Valida as colunas obrigatórias e as datas, aplica deduplicação e as regras
    causais de ``preparar_valores_ausentes_vendas``, e cria calendário e
    atributos históricos da quantidade vendida (lags e janelas móveis).
    Descarta observações anteriores a 2025 antes de criar atributos temporais,
    pois 2024 tem cobertura parcial e comportamento não comparável à série
    observada de 2025 e 2026.

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
    # Ao contrário da EDA, o modelo não pode usar estatísticas de semanas futuras.
    df = preparar_valores_ausentes_vendas(df, causal_imputation=True)

    iso_calendar = df["semana"].dt.isocalendar()
    # Semana ISO temporária, usada apenas para criar as variáveis cíclicas abaixo.
    semana_iso = iso_calendar.week.astype(int)

    df = df.sort_values(["sku_id", "canal", "semana"]).copy()

    series_groups = df.groupby(["sku_id", "canal"], sort=False)
    sold_quantity = series_groups["qt_vendida"]
    # Vendas anteriores ajudam o modelo a reconhecer o nível recente da série,
    # sem incluir a semana que será prevista.
    for lag in [1, 2, 4, 8]:
        df[f"venda_defasada_{lag}"] = sold_quantity.shift(lag)

    # Média das quatro semanas anteriores para representar o nível recente de venda.
    df["media_movel_4"] = sold_quantity.transform(
        lambda values: values.shift(1).rolling(4).mean()
    )

    # Média das oito semanas anteriores para representar uma tendência mais estável.
    df["media_movel_8"] = sold_quantity.transform(
        lambda values: values.shift(1).rolling(8).mean()
    )

    # Desvio padrão das quatro semanas anteriores para medir a variação recente das vendas.
    df["desvio_padrao_movel_4"] = sold_quantity.transform(
        lambda values: values.shift(1).rolling(4).std()
    )

    # Quantidade de ruptura na semana anterior, como sinal de venda possivelmente limitada.
    previous_stockout = series_groups["qt_ruptura"].shift(1)
    df["ruptura_defasada_1"] = previous_stockout

    # Indicador binário de ruptura na semana anterior; preserva ausência como NaN.
    df["flag_ruptura_defasada_1"] = np.where(
        previous_stockout.isna(),
        np.nan,
        previous_stockout.gt(0).astype(int),
    )

    # Seno da semana do ano para representar a sazonalidade sem salto no fim do ano.
    df["semana_seno"] = np.sin(2 * np.pi * semana_iso / 52)

    # Cosseno complementar ao seno para identificar a posição no ciclo anual.
    df["semana_cosseno"] = np.cos(2 * np.pi * semana_iso / 52)
    return df

## Opcional: Executa para verificar se o carregamento e preparação funcionam corretamente e exibe o DataFrame resultante usado no modelo. 
# Remova ou comente esta parte se não for necessária.
dados = carregar_dados_vendas()
df_modelo = preparar_dados_modelo(dados)
df_modelo
