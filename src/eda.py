# EDA

## Importações
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
if __package__:
    from .data_utils import (
        EDA_OUTPUT_DIR,
        NUMERIC_COLUMNS,
        carregar_dados_vendas,
        preparar_valores_ausentes_vendas,
        remover_duplicatas_vendas,
    )
else:
    from data_utils import (
        EDA_OUTPUT_DIR,
        NUMERIC_COLUMNS,
        carregar_dados_vendas,
        preparar_valores_ausentes_vendas,
        remover_duplicatas_vendas,
    )

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
sns.set_theme(style='whitegrid', palette='viridis', context = 'talk')
COR_FUNDO = '#f5f1eb'
plt.rcParams.update({
    'figure.facecolor': COR_FUNDO,
    'axes.facecolor': COR_FUNDO,
    'savefig.facecolor': COR_FUNDO,
    'axes.edgecolor': '#817b72',
    'axes.labelcolor': '#393630',
    'text.color': '#393630',
    'xtick.color': '#514d46',
    'ytick.color': '#514d46',
    'grid.color': '#d8d1c7',
    'grid.alpha': 0.65,
    'axes.titleweight': 'bold',
    'axes.titlepad': 12,
})

FIGDIR = EDA_OUTPUT_DIR

def save_show(fig, name):
    """Salva toda figura."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGDIR / f'{name}.png',
        dpi=140,
        bbox_inches='tight',
        facecolor=COR_FUNDO,
    )
    plt.show()
    plt.close(fig)

#-------------------------------------------------------------------------------#
# Qual é o shape, os tipos, o intervalo temporal e a granularidade real?
df = carregar_dados_vendas()

print('=' * 70)
print(f'SHAPE: {df.shape[0]:,} linhas × {df.shape[1]} colunas')
print('=' * 70)

print('\n--- DTYPES ---')
print(df.dtypes)

print('\n--- HEAD(8) ---')
print(df.head(8))

print('\n--- TAIL(8) ---')
print(df.tail(8))

print('\n--- ESTATÍSTICAS DESCRITIVAS (numéricas) ---')
print(df.describe().T.round(2))

print('\n--- ESTATÍSTICAS DESCRITIVAS (object) ---')
print(df.describe(include='object').T)

print('\n--- CARDINALIDADE (nunique por coluna) ---')
print(df.nunique().sort_values(ascending=False))

"""
COMENTÁRIOS:

SHAPE: 3,736 linhas e 14 colunas
qt_vendida com mínimo negativo → conferir coerência com receita e preço médio
receita e preco_medio com possíveis outliers → verificar se são erros de digitação ou promoções legítimas
deconto_pct com valores > 100% → Não faz sentido, provavelmente erro na fonte de dados
Há 20 skus, 3 categorias, 8 subcategorias e 2 canais

"""

#-------------------------------------------------------------------------------#

# Quantos e onde estão os dados faltantes(caso existam)?
missing = (df.isna().sum()
             .to_frame('qtd_missing'))
missing['pct'] = (missing['qtd_missing'] / len(df) * 100).round(2)
missing = missing[missing['qtd_missing'] > 0].sort_values('pct', ascending=False)
print('--- MISSING POR COLUNA (ANTES DE QUALQUER TRATAMENTO) ---')
print(missing)


## Visualização: heatmap de missing
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df.isna(), cbar=False, cmap='gray_r', ax=ax, yticklabels=False)
ax.set_title('Mapa de valores ausentes (linha = observação, coluna = campo)')
save_show(fig, '01_dados_faltantes')
plt.show()

"""
COMENTÁRIOS
- Há valores ausentes em 7 colunas, concentrados principalmente em margem_pct.
"""

#-------------------------------------------------------------------------------#
# Qual é o formato da coluna semana(data)? - Pergunta importante para o modelo predititvo, pois é um campo de data que pode ser parseado de várias formas diferentes.

## Passo 1: ver como strings cruas — NÃO parsear ainda
sample = df['semana'].drop_duplicates().sample(30, random_state=42).tolist()
print('--- AMOSTRA DE VALORES ÚNICOS DA COLUNA `semana` (como string) ---')
for s in sorted(sample):
    print(f'  {s!r}')

## Passo 2: contar padrões
df['_fmt'] = np.where(df['semana'].astype(str).str.contains('-', na=False),
                      'ISO (YYYY-MM-DD)', 'BR (DD/MM/YYYY)')
print('\n--- DISTRIBUIÇÃO DE FORMATOS NA COLUNA `semana` ---')
print(df['_fmt'].value_counts(dropna=False))

print('\n--- EXEMPLOS DE CADA FORMATO ---')
for fmt, g in df.groupby('_fmt'):
    print(f'  {fmt}: {g["semana"].drop_duplicates().head(5).tolist()}')

"""
COMENTÁRIOS
- Há dois formatos de datas, 3 linhas no formato popular no Brasil (DD/MM/YYYY) e 
os demais no formato ISO (YYYY-MM-DD), será necessário ajuste

"""

## Passo 3: Trata casos identificados 
df['semana_dt'] = pd.to_datetime(df['semana'], format='mixed', dayfirst=True, errors='coerce')
df['semana'] = df['semana_dt']

n_falhas = df['semana_dt'].isna().sum()
print(f'\nFalhas de parsing após to_datetime: {n_falhas}')
assert n_falhas == 0, 'Investigar linhas com data irrecuperável'

print('\n--- INTERVALO TEMPORAL COBERTO ---')
print(f'De: {df["semana_dt"].min().date()}')
print(f'Até: {df["semana_dt"].max().date()}')
print(f'Nº de semanas ISO distintas: {df["semana_dt"].dt.to_period("W").nunique()}')

"""
COMENTÁRIOS
- Após o parsing, o intervalo temporal vai de Fevereiro de 2024 a Dezembro de 2026, 
cobrindo 94 distintas.

"""

#-------------------------------------------------------------------------------#
## A chave (SKU, canal, semana) é única? Se não, as duplicatas são idênticas ou "quase"?

## Duplicatas exatas
print('\n')
n_exatas = int(df.duplicated().sum())
print(f'Duplicatas EXATAS: {n_exatas}')

## Identificar grupos repetidos pela chave nos dados originais, antes de deduplicar
chave = ['sku_id', 'canal', 'semana_dt']
dups_chave = (
    df.groupby(chave, dropna=False)
    .size()
    .reset_index(name='n')
    .query('n > 1')
)
print(f'\nChaves (sku_id, canal, semana) com MAIS DE UMA LINHA: {len(dups_chave)}')
print(dups_chave.sort_values('n', ascending=False).head(10))

## Classificar cada chave repetida comparando suas linhas originais
if len(dups_chave) > 0:
    merged = df.merge(dups_chave[chave], on=chave, how='inner')
    colunas_comparacao = [
        coluna for coluna in df.columns
        if coluna not in chave + ['semana', '_fmt']
    ]
    variacao = (
        merged.groupby(chave, dropna=False)[colunas_comparacao]
              .nunique(dropna=False)
              .reset_index()
    )
    variacao['tipo'] = np.where(variacao[colunas_comparacao].max(axis=1) > 1,
                                'QUASE (valores diferem)', 'EXATA (valores iguais)')
    print('\n--- DAS DUPLICATAS DE CHAVE, QUANTAS SÃO EXATAS vs QUASE ---')
    print(variacao['tipo'].value_counts())

    print('\n--- EXEMPLOS DE DUPLICATAS QUASE ---')
    chaves_quase = variacao[variacao['tipo'].str.startswith('QUASE')][chave].head(5)
    print(df.merge(chaves_quase, on=chave, how='inner')
            .sort_values(chave)
            [chave + ['qt_vendida','qt_ruptura','receita','preco_medio','desconto_pct']]
            .round(2))
"""
Comentário: Após a análise, existem duplicatas exatas e "quase" (valores diferentes nos demais campos).
As duplicatas exatas podem ser removidas sem perda de informação.
As duplicatas "quase" foram tratatadas priorizando a completude e proximidade às medianas, 
preservando a ordem original em caso de empates.
Essa alternativa foi escolhida visando uma solução escalável e automatizada, 
sem necessidade de intervenção manual para cada caso. Tratamento em 'data_utils.py'

"""
## Após a análise, remover exatas; para as "quase", manter uma linha observada por chave.
## Prioriza completude e proximidade às medianas; empates preservam a ordem original.
## Mais detalhes em data_utils/remover_duplicatas_vendas()
df = remover_duplicatas_vendas(df, chave)
num_cols = NUMERIC_COLUMNS
print(f'\nApós deduplicação, shape: {df.shape}')

# Diagnosticar valores inválidos antes de aplicar as correções
neg = pd.DataFrame({
    'n_negativos': [(df[c] < 0).sum() for c in num_cols],
    'min':         [df[c].min() for c in num_cols],
}, index=num_cols)
print('--- VALORES NEGATIVOS POR COLUNA ---')
print(neg)

print('\n--- EXEMPLOS DE QUANTIDADE NEGATIVA ---')
print(df[df['qt_vendida'] < 0]
        .sort_values('qt_vendida')
        [['sku_id','canal','semana_dt','qt_vendida','qt_ruptura', 'preco_medio', 'receita']]
        .head(5))
"""
COMENTÁRIOS:
- A coluna qt_vendida tem  3 valores negativos, suspeita-se que sejam erros na fonte de dados, visto que
a fonte de dados não aparenta incluir valores de "devolução"(apenas 3 registros).
- Além disso, a coluna receita está positiva e é igual ao qt_vendida(positiva) multiplicada pelo preço médio. 
Nesse contexto, optou-se por transformar a quantidade como qt_vendida em valores absolutos(positivos), para manter 
a coerência com a receita e o preço médio. Tratamento em 'data_utils.py'
"""

print('\n--- DESCONTOS ACIMA DE 100% (INVÁLIDOS) ---')
descontos_invalidos = df.loc[
    df['desconto_pct'] > 100,
    [
        'sku_id', 'canal', 'semana_dt', 'desconto_pct',
        'receita', 'preco_medio', 'margem_pct',
    ],
]
print(descontos_invalidos.sort_values('desconto_pct', ascending=False))
n_descontos_invalidos = len(descontos_invalidos)

"""
COMENTÁRIOS: Descontos superiores a 100% não fazem sentido, 
provavelmente erro na fonte de dados, dessa forma optou-se por retirar da análise e do modelo
Tratamento em 'data_utils.py'
- 
"""
# Avaliação dos outliers de preço médio observados na análise descritiva

# Distribuição de preço médio POR SKU (boxplot) — evidencia outliers visuais
fig, ax = plt.subplots(figsize=(16, 5))
sns.boxplot(data=df, x='sku_id', y='preco_medio', ax=ax, color='steelblue')
ax.set_title('Distribuição de preço médio por SKU')
ax.set_xlabel(''); ax.set_ylabel('Preço médio (R$)')
plt.xticks(rotation=45)
save_show(fig, '02_preco_por_sku_boxplot')


"""
COMENTÁRIO: Nota-se que há 3 skus com valores "muito fora da curva", acredita-se que houve
erro de inputação de preço médio vide a distribuição. 
Vamos aplicar uma correção, usando a premissa já usada: preco_medio = receita/qt_vendida 
e avaliar se o preço ajustado se encaixa na distribuição do preço do SKU.
"""

mediana_preco_sku = df.groupby('sku_id')['preco_medio'].transform('median')
outliers_preco = df.loc[
    df['preco_medio'] > 5 * mediana_preco_sku,
    [
        'sku_id', 'canal', 'semana_dt', 'preco_medio',
        'receita', 'qt_vendida',
    ],
].copy()
outliers_preco['mediana_preco_sku'] = df.loc[
    outliers_preco.index, 'sku_id'
].map(df.groupby('sku_id')['preco_medio'].median())
outliers_preco['preco_calculado'] = (
    outliers_preco['receita']
    / outliers_preco['qt_vendida'].abs().replace(0, np.nan)
)
outliers_preco['preco_calculado_mediana'] = (
    outliers_preco['preco_calculado']
    / outliers_preco['mediana_preco_sku']
)
outliers_preco['limite_5x_mediana'] = (
    5 * outliers_preco['mediana_preco_sku']
)
outliers_preco['correcao_aceita'] = (
    outliers_preco['preco_calculado'].gt(0)
    & outliers_preco['preco_calculado'].le(
        outliers_preco['limite_5x_mediana']
    )
)
print('\n--- AVALIAÇÃO DA CORREÇÃO DOS OUTLIERS DE PREÇO ---')
print(
    outliers_preco.sort_values('preco_medio', ascending=False)
    .round(2)
    .to_string(index=False)
)

"""
COMENTÁRIO: De fato, o preço calculado a partir da receita e quantidade vendida é coerente com a mediana do SKU,
Além disso, nota-se que os preços calculados são 10x superiores 
aos presentes na fontes de dados, o que indica erro de inputação da vírgula na casa decimal.
Tratamento em 'data_utils.py'
"""

#-------------------------------------------------------------------------------#
##### TRATAMENTO FINAL ANTES DA ANÁLISE DE INSIGHTS #####
"""
COMENTÁRIOS:
- As relações receita = qt_vendida * preco_medio permitem recuperar um campo ausente
quando os outros dois estão disponíveis e válidos. Caso dois desses campos estejam ausentes (diferente de 0) 
assumiu-se como linha não válida para uso
- A EDA encontrou linhas sem receita e sem quantidade vendida: sem nenhum dos dois
  valores não é possível determinar a demanda nem recuperar a receita, então essas
  linhas são removidas.
- Quando qt_vendida é zero e receita está ausente, o registro indica ausência de
  venda; preencher receita com zero preserva essa observação sem inventar demanda.
- Preço ausente em registros com quantidade e receita iguais a zero é compatível
  com ausência de venda: sem transação não há preço observado.
- Para a subcategoria também pode ser preenchida quando SKU, categoria e canal 
apontam para uma única subcategoria.
- Mantemos qt_ruptura ausente porque ausência não significa zero. 
- Para os demais campos: margem_pct e desconto_pct considerou-se o imput 
mediana avaliando SKU-Canal para linhas com Receita > 0, visto que desconto aplicado sem venda não faz sentido
Tratamento em 'data_utils.py'
"""

# Aplicar correções
casos_receita_quantidade = pd.Series(
    {
        'receita e qt_vendida ausentes': int(
            (df['receita'].isna() & df['qt_vendida'].isna()).sum()
        ),
        'receita ausente com qt_vendida igual a zero': int(
            (df['receita'].isna() & df['qt_vendida'].eq(0)).sum()
        ),
        'preço ausente com receita e qt_vendida iguais a zero': int(
            (
                df['preco_medio'].isna()
                & df['receita'].eq(0)
                & df['qt_vendida'].eq(0)
            ).sum()
        ),
    },
    name='quantidade de linhas',
)
print('\n--- DIAGNÓSTICO DOS CASOS DE RECEITA, QUANTIDADE E PREÇO ---')
print(casos_receita_quantidade)

missing_antes = df.isna().sum()
subcategorias_antes = df['subcategoria_id'].isna().sum()
# Na EDA exploratória, usamos toda a série para descrever/imputar distribuições;
# esse modo não é apropriado para prever semanas futuras.
df = preparar_valores_ausentes_vendas(df, causal_imputation=False)
print(
    '\nQuantidade vendida negativa após ajuste:',
    int(df['qt_vendida'].lt(0).sum()),
)
print(
    'Descontos acima de 100% após ajuste:',
    int(df['desconto_pct'].gt(100).sum()),
    f'(identificados antes: {n_descontos_invalidos})',
)
print('\n--- MISSING APÓS AS TRANSFORMAÇÕES ---')
missing_apos = (
    df.drop(columns='qt_vendida_raw')
      .isna()
      .sum()
      .to_frame('qtd_missing')
)
missing_apos['pct'] = (missing_apos['qtd_missing'] / len(df) * 100).round(2)
missing_apos = (
   missing_apos[missing_apos['qtd_missing'] > 0]
   .sort_values('pct', ascending=False)
)
print(missing_apos)

imputacoes = missing_antes - df.drop(columns='qt_vendida_raw').isna().sum()
print('\n--- MISSING IMPUTADOS POR COLUNA ---')
print(imputacoes[imputacoes > 0])
print(
    'Subcategorias imputadas por mapeamento único:',
    int(subcategorias_antes - df['subcategoria_id'].isna().sum()),
)
#-------------------------------------------------------------------------------#
### -- Análise dos Dados -- ###

print ('\n--- ANÁLISE DE INSIGHTS ---')

# Ruptura — frequência, padrão temporal, relação com a venda
 ## Perguntas:
 ### Quais SKUs com maior ruptura no periodo?
 ### Qual o valor da ruptura em R$ perdida?
 ### A ruptura é episódica ou sistemática?
 ### Quando ela ocorre (sazonalidade)?
 ### Ela reduz a venda observada?

## Estatísticas de ruptura por SKU
teve_ruptura = df['qt_ruptura'].gt(0).where(df['qt_ruptura'].notna())
rupt_stats = (
    df.assign(
        teve_ruptura=teve_ruptura,
        valor_ruptura_rs=df['qt_ruptura'] * df['preco_medio'],
    )
      .groupby('sku_id')
      .agg(semanas=('qt_ruptura', 'size'),
           semanas_com_dado=('qt_ruptura', 'count'),
           taxa_ruptura=('teve_ruptura', 'mean'),
           ruptura_media=('qt_ruptura', 'mean'),
           ruptura_max=('qt_ruptura', 'max'),
           ruptura_total_rs=('valor_ruptura_rs', lambda values: values.sum(min_count=1)))
      .sort_values('ruptura_total_rs', ascending=False)
)
print('--- RUPTURA POR SKU: FREQUÊNCIA, VOLUME E VALOR ESTIMADO ---')
print(rupt_stats.round(2))

## Visualização: valor estimado da ruptura por SKU
fig, ax = plt.subplots(figsize=(14, 6))
ruptura_plot = rupt_stats.sort_values('ruptura_total_rs', ascending=True)
sns.barplot(
    x='ruptura_total_rs',
    y=ruptura_plot.index,
    data=ruptura_plot.reset_index(),
    ax=ax,
    color='#c66b4e',
)
ax.set_title('Valor estimado da ruptura por SKU')
ax.set_xlabel('Valor estimado da ruptura (R$)')
ax.set_ylabel('')
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda value, _: f'R$ {value / 1000:,.0f} mil')
)
ax.grid(axis='y', visible=False)
ax.grid(axis='x', alpha=0.5)
ax.spines[['top', 'right', 'left']].set_visible(False)
top_valor = ruptura_plot['ruptura_total_rs'].iloc[-1]
ax.text(
    top_valor,
    len(ruptura_plot) - 1,
    f'  R$ {top_valor:,.0f}',
    va='center',
    fontweight='bold',
)
save_show(fig, '03_valor_ruptura_por_sku')

"""
COMENTÁRIO: O SKU_01 concentra o maior valor estimado (quase 9x superior ao segundo colocado, SKU_10), e com 15% das semanas
com ruptura no período avaliado, portanto é um SKU prioritário e provalmente o TOP em vendas.
"""

## Comparação: distribuição de qt_vendida nas semanas COM vs SEM ruptura
fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(data=df.assign(ruptura_grupo=np.select(
                   [df['qt_ruptura'].isna(), df['qt_ruptura'].gt(0)],
                   ['sem dado', 'com ruptura'],
                   default='sem ruptura',
               )),
               x='ruptura_grupo', y='qt_vendida', ax=ax, split=False, inner='quart')
ax.set_title('Vendas observadas por situação de ruptura')
ax.set_xlabel('')
ax.set_ylabel('Unidades vendidas na semana')
ax.spines[['top', 'right']].set_visible(False)
save_show(fig, '04_violin_ruptura_vs_venda')

# Comparação numérica
comp = (
    df.assign(com_ruptura=teve_ruptura)
      .groupby('com_ruptura')['qt_vendida']
      .agg(['mean','median','count'])
)
print('\n--- MÉDIA/MEDIANA DE qt_vendida: COM vs SEM ruptura ---')
print(comp.round(2))

"""
COMENTÁRIO: a distribuição mostra vendas com e sem ruptura, nota-se que a disrtbuição em ambos os cenários
são semelhantes. No cenário Sem ruptura, há semanas com maior pico de vendas, embora a média de vendas Com ruptura tenha sido 9% superior
à semanas Sem ruputura, o que pode indicar eventos promocionais, sazonalidade que aumentaram o volume de vendas em algumas semanas, descolando a distribuição média para cima
"""
#-------------------------------------------------------------------------------#

# A Venda e a ruptura tem padrão de semana do ano? É uniforme entre SKUs/canais?

# Transformar a semana em semana ISO (segunda a domingo)
df['semana_iso'] = df['semana_dt'].dt.isocalendar().week.astype(int)
df['mes'] = df['semana_dt'].dt.month
df['trimestre'] = df['semana_dt'].dt.quarter

# Ruptura e demanda real totais por semana ISO
ruptura_demanda_semana = (
    df.groupby('semana_iso')[['qt_ruptura', 'qt_vendida']]
      .sum()
      .reset_index()
)
fig, ax = plt.subplots(figsize=(16, 5))
sns.lineplot(
    data=ruptura_demanda_semana,
    x='semana_iso',
    y='qt_vendida',
    ax=ax,
    marker='o',
    color='#3f7180',
    linewidth=2.5,
    label='Demanda',
)
sns.lineplot(
    data=ruptura_demanda_semana,
    x='semana_iso',
    y='qt_ruptura',
    ax=ax,
    marker='o',
    color='#c66b4e',
    linewidth=2.5,
    label='Ruptura',
)
ax.set_title('Venda e Ruptura por semana')
ax.set_xlabel('Semana ISO'); ax.set_ylabel('Unidades totais (escala log)')
ax.set_yscale('log')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
save_show(fig, '05_ruptura_demanda_semana')


"""
COMENTÁRIO: Observa-se que tanto a Venda quanto a Ruptura tem comportamentos semelhantes.
E como a sazonalidade de fim de ano influencia na ruptura, é possível ver uma tendência de crescimento
nas últimas semanas do ano

Para as análises de demanda a seguir, optou-se por avaliar a Demanda real, somando os valores
de Valor vendido e Quantidade rupturada (deixou de ser vendido) para se ter uma melhor leitura da real necessidade dos consumidores
"""

## Criar coluna de demanda real (qt_vendida + qt_ruptura)
df['demanda_real'] = df['qt_vendida'] + df['qt_ruptura']

# Sazonalidade agregada — Total por semana ISO
saz_geral = df.groupby('semana_iso')['demanda_real'].sum()
fig, ax = plt.subplots(figsize=(16, 5))
sns.lineplot(
    x=saz_geral.index,
    y=saz_geral.values,
    ax=ax,
    marker='o',
    color='#3f7180',
    linewidth=2.5,
)
ax.axvspan(48, 52, alpha=0.14, color='#c66b4e', label='Fim de ano')
ax.axvspan(2, 13, alpha=0.14, color='#d5a64a', label='Pós-festas')
ax.set_title('A demanda real varia ao longo do ano')
ax.set_xlabel('Semana ISO'); ax.set_ylabel('Demanda total (unidades)')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
save_show(fig, '06_sazonalidade_semana')

# Sazonalidade por semana ISO e canal
saz_canal = (
    df.groupby(['canal', 'semana_iso'])['demanda_real']
      .sum()
      .reset_index()
)
fig, ax = plt.subplots(figsize=(16, 5))
sns.lineplot(
    data=saz_canal,
    x='semana_iso',
    y='demanda_real',
    hue='canal',
    ax=ax,
    marker='o',
    linewidth=2.5,
)
ax.axvspan(48, 52, alpha=0.14, color='#c66b4e', label='Fim de ano')
ax.axvspan(2, 13, alpha=0.14, color='#d5a64a', label='Pós-festas')
ax.set_title('A demanda real por canal varia ao longo do ano')
ax.set_xlabel('Semana ISO'); ax.set_ylabel('Demanda total (unidades)')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(title='Canal / período')
save_show(fig, '07_sazonalidade_semana_por_canal')

"""
COMENTÁRIO: os trechos sombreados destacam a sazonalidade, além disso observa-se como
no fim do ano ocorrem picos e no ínicio do ano a demanda é mais baixa, após a semana 20
observa-se relativa estabilidade.
"""

# Sazonalidade por mês
fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(
    data=df,
    x='mes',
    y='demanda_real',
    ax=ax,
    color='#7da3a0',
    showfliers=False,
)
ax.set_title('A distribuição mensal evidencia nível e dispersão da demanda')
ax.set_xlabel('Mês'); ax.set_ylabel('Demanda semanal (un.)')
ax.spines[['top', 'right']].set_visible(False)
save_show(fig, '08_sazonalidade_mes')

"""
COMENTÁRIO: Os meses de Novembro e Dezembro são os mais desafiadores devido a alta variabilidade.
Abril e Maio tem semelhanças com o fim do ano, os demais meses comportamento semelhante.
Obs: Os pontos extremos foram ocultados apenas nesta
visualização para facilitar a comparação; eles continuam nos dados e nas análises.
"""



# Sazonalidade por categoria
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
for ax, cat in zip(axes, sorted(df['categoria_id'].unique())):
    d = df[df['categoria_id'] == cat].groupby('semana_iso')['demanda_real'].sum().reset_index()
    sns.lineplot(
        data=d, x='semana_iso', y='demanda_real', ax=ax,
        color='#3f7180', linewidth=2,
    )
    ax.set_title(f'Categoria {cat}')
    ax.set_xlabel('Semana ISO'); ax.set_ylabel('Demanda total')
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_show(fig, '09_sazonalidade_por_categoria')

"""
COMENTÁRIO: Observa-se que a categoria B possui maior demanda, mas todas as subcategorias apresentam
comportamentos semelhantes ao longo do ano, como observado na curva de demanda geral. Interesse notar que todas as categorias
possuem picos proximos a semana 15 e semana 37, possivelmente por promoções e/ou datas comemorativas
Obs: Os painéis mantêm escalas independentes para tornar legíveis as
oscilações internas de cada categoria. 
"""

#-------------------------------------------------------------------------------#

# Os SKUs têm tendência e comportamentos semelhantes? Quais são atípicos?

## Uma linha por SKU, agregando os canais
df['ano'] = df['semana_dt'].dt.year
resumo_anual_demanda = df.groupby('ano')['demanda_real'].agg(
    semanas_com_demanda='count',
    media='mean',
    mediana='median',
    desvio_padrao='std',
)
print('\n--- COBERTURA E DEMANDA POR ANO ---')
print(df.groupby('ano')['semana_dt'].agg(inicio='min', fim='max', semanas='nunique'))
print(resumo_anual_demanda.round(2))

demanda_sku_semana = (
    df.groupby(['sku_id', 'semana_dt'], as_index=False)['demanda_real']
      .mean()
)
skus = sorted(demanda_sku_semana['sku_id'].unique())

fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 12), sharex=False)
for ax, sku in zip(axes.flat, skus):
    d = demanda_sku_semana[demanda_sku_semana['sku_id'] == sku]
    ax.axvspan(
        pd.Timestamp('2024-01-01'),
        pd.Timestamp('2025-01-01'),
        color='#d5a64a',
        alpha=0.12,
        zorder=0,
    )
    ax.axvline(pd.Timestamp('2025-01-01'), color='#8c5c3f', linewidth=0.8)
    sns.lineplot(data=d, x='semana_dt', y='demanda_real',
                 ax=ax, color='#3f7180', linewidth=1.4)
    ax.set_title(sku, fontsize=10)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=7, rotation=30)
    ax.spines[['top', 'right']].set_visible(False)
plt.suptitle(
    'Demanda semanal por SKU',
    y=1.02,
    fontweight='bold',
)
plt.tight_layout()
save_show(fig, '10_demanda_por_sku')

"""
COMENTÁRIO: Destaques obervados: 
- SKU 03 e SKU 15 apresentam tendência de crescimento de demanda,
- SKU 02 e 13 aparentam ser lançamentos dado o horizonte temporal de vendas
- A demanda de 2024 para todos os SKUs com vendas nesse período apresenta comportamento atípico 
vs 2025 e 2026

Obs: 2024 contém somente 11 semanas observadas,
enquanto 2025 tem 52 e 2026 tem 43. Além da cobertura parcial, a média e a
dispersão da demanda de 2024 diferem das observadas em 2025 e 2026. A janela não é
comparável e pode distorcer o aprendizado temporal; por isso, 2024 será removido
antes da criação de lags e demais atributos do modelo preditivo. O sombreado
destaca essa janela; a linha vertical marca o início do período utilizado.
"""

#-------------------------------------------------------------------------------#
# Os picos de demanda se concentram nas mesmas semanas para todos os SKUs, ou cada um tem seu calendário?

pivot = df.pivot_table(index='sku_id', columns='semana_iso',
                       values='demanda_real', aggfunc='mean')
fig, ax = plt.subplots(figsize=(22, 8))
sns.heatmap(
    pivot,
    cmap='YlOrBr',
    cbar_kws={'label': 'Demanda média (unidades)'},
    ax=ax,
    linewidths=0.2,
    linecolor=COR_FUNDO,
)
ax.set_title('Semanas de pico de demanda por SKU')
ax.set_xlabel('Semana ISO'); ax.set_ylabel('')
save_show(fig, '11_heatmap_sku_semana')

"""
COMENTÁRIO: Complementar à visão anterior, nota-se que os picos são semelhantes entre
todos os SKUs, destaque para os SKUs 01 e 10 que apresentam os maiores picos, sobretudo na semana 19
e semanas dos meses de Novembro e Dezembro
"""
#-------------------------------------------------------------------------------#
# As mecânicas são usadas com frequência semelhante?
# Qual mecânica gera mais lift (crescimento incremental) em cada SKU?

df['mecanica'] = np.select(
    [df['flag_desconto_direto'].astype(bool),
     df['flag_combo'].astype(bool),
     df['flag_pague_e_leve'].astype(bool)],
    ['desconto_direto', 'combo', 'pague_e_leve'],
    default='sem_promo',
)

print('--- FREQUÊNCIA DE MECÂNICAS ---')
print(df['mecanica'].value_counts(normalize=True).round(3) * 100)

# Boxplot de desconto_pct por mecânica
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df[df['mecanica'] != 'sem_promo'],
            x='mecanica', y='desconto_pct', ax=ax,
            color='#d5a64a', showfliers=False)
ax.set_title('A profundidade do desconto varia por mecânica promocional')
ax.set_xlabel(''); ax.set_ylabel('Desconto (%)')
ax.spines[['top', 'right']].set_visible(False)
save_show(fig, '12_desconto_por_mecanica')

"""
COMENTÁRIO: De forma geral, as mecânicas de Desconto Direto e Combo foram as mais utlizdas,
10 e 9%, respecitvamente. Embora a mecânica Pague e Leve seja a menos frequente, ela é a mais "agressiva",
com desconto mediano superando os 30%. Além disso, aparentemente há alguma politica de desconto máximo de até 45%
de demanda.
"""

# Boxplot de demanda por mecânica — pequenos múltiplos por SKU (top 5 volume)
top_skus = df.groupby('sku_id')['demanda_real'].mean().nlargest(5).index
fig, axes = plt.subplots(1, 5, figsize=(16, 8))
for ax, sku in zip(axes.flat, top_skus):
    d = df[df['sku_id'] == sku]
    sns.boxplot(
        data=d,
        x='mecanica',
        y='demanda_real',
        ax=ax,
        color='#7da3a0',
        showfliers=False,
        order=['sem_promo', 'desconto_direto', 'combo', 'pague_e_leve'],
    )
    ax.set_title(sku, fontsize=10)
    ax.set_xlabel(''); ax.set_ylabel('Demanda (un.)')
    ax.tick_params(axis='x', labelsize=8, rotation=20)
    ax.spines[['top', 'right']].set_visible(False)
plt.suptitle('Demanda por mecânica nos 5 SKUs de maior volume', y=1.02, fontweight='bold')
plt.tight_layout()
save_show(fig, '13_boxplot_demanda_por_mecanica')

"""
COMENTÁRIO: Com base no que vimos até aqui, o SKU_01 aparenta ser o "Hero" da marca, visto que
a demanda mediana varia de cerca de 225 unidades/semana para aproximadamente 300 unidadades/semana com o Desconto Direto.
A mecância de Combo aparenta ser bem efetiva para esse SKU, com a demanda mínima próxima a mediana não promocional
Para o SKU_10, a mecânica que indica alguma efetividade é a Pague e Leve, com cerca 75% da distribuição da demanda superior a 
mediana da distribuição não promocional
Analisando os demais SKUs do TOP5, observa-se que cada SKU possui suas e características
OBS: Embora é possivel observar diferenças claras entre mecânicas, para um diagnóstico mais conclusivo 
e estatísticamente confiável, Testes A/B podem comprovar o fenômeno aqui observado e indicar qual a 
mecânica apropriada para cada SKU.



"""

#-------------------------------------------------------------------------------#
# Como se distribuem preço e venda em escala log por SKU?

# Scatter log-log por SKU, todos os canais, restrito a semanas SEM ruptura
todos_canais_ok = df[
    (df['qt_ruptura'] == 0)
    & (df['qt_vendida'] > 0)
    & (df['preco_medio'] > 0)
].copy()
fig, axes = plt.subplots(4, 5, figsize=(20, 12), sharex=False)
for ax, sku in zip(axes.flat, skus):
    d = todos_canais_ok[todos_canais_ok['sku_id'] == sku].copy()
    if len(d) < 5:
        ax.text(0.5, 0.5, 'poucos pontos', ha='center', va='center')
        ax.set_title(sku); continue
    d['ln_q'] = np.log(d['qt_vendida'])
    d['ln_p'] = np.log(d['preco_medio'])
    sns.regplot(
        data=d,
        x='ln_p',
        y='ln_q',
        ax=ax,
        scatter_kws={'alpha': 0.45, 's': 18, 'color': '#3f7180'},
        line_kws={'color': '#c66b4e', 'linewidth': 1.8},
    )
    ax.set_title(sku, fontsize=10)
    ax.set_xlabel(''); ax.set_ylabel('')
fig.supxlabel('Log do preço médio')
fig.supylabel('Log da quantidade vendida (un.)')
plt.suptitle(
    'Relação preço–venda em escala log — todos os canais, semanas sem ruptura',
    y=1.02,
    fontweight='bold',
)
plt.tight_layout(rect=[0.04, 0.04, 1, 0.96])
save_show(fig, '14_loglog_elasticidade')

"""
COMENTÁRIO: A reta é um ajuste linear simples em escala log-log, incluído como
referência visual. Os Desvios sistemáticos dos pontos em relação à reta sugeream uma relação não-linear
o que indica uma elasticidade que não é constante para os SKUs analisados no período
Para uma estimativa de elasticidade para ajudar a entnder a relação preço-demanda, modelos mais robustos e um
estudo mais apronfudado seria necessário e dessa forma não faz parte do escopo desta análise
"""

#-------------------------------------------------------------------------------#
# SKUs da mesma subcategoria têm demanda correlacionada? Promoção de um afeta o outro?

# Correlação de demanda entre SKUs da mesma subcategoria, agregando canais
for subcat, g in df.groupby('subcategoria_id'):
    if g['sku_id'].nunique() < 2: continue
    pivot = g.pivot_table(
        index='semana_dt',
        columns='sku_id',
        values='demanda_real',
        aggfunc='mean',
    )
    if pivot.shape[0] < 20 or pivot.shape[1] < 2: continue
    corr = pivot.corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
        linewidths=0.4,
        linecolor=COR_FUNDO,
        cbar_kws={'label': 'Correlação de Pearson'},
    )
    ax.set_title(f'Correlação de Demanda entre SKUs — {subcat}')
    save_show(fig, f'16_corr_skus_{subcat}')

"""
COMENTÁRIO: correlação positiva indica semanas em que as demandas entre os SKUs tendem a
subir ou cair juntas; correlação negativa indica movimentos opostos. Isso não
estabelece canibalização nem efeito de promoção, pois calendário e fatores
comuns podem explicar a associação.
De todo modo, no histórico avaliado:
CAT_A_SUB1: SKU_08 e SKU_04 mostraram razoaveis movimentos opostos, um aprofundamento
poderia validar possível efeito de canibalização, assim como observado em CAT_A_SUB2 
para os SKUs 12 e 05
Entre todas as Subcategorias, a CAT_A SUB3 foi a que mostrou mais indícios de possível canibalização
para os SKUS 14 e 17 (R Pearson = -0.44). Analogamente, os SKUs 15 e 11 da CAT_C_SUB1 
tiveram a maior correlação positiva (0,80). Possivelmente, SKUs de uso complementar pelo consumidor

"""

#-------------------------------------------------------------------------------#
# Os dois canais têm a mesma distribuição de demanda, desconto, ruptura?

# Estatísticas por canal
canal_stats = df.groupby('canal').agg(
    n=('demanda_real', 'count'),
    demanda_media=('demanda_real', 'mean'),
    demanda_mediana=('demanda_real', 'median'),
    preco_medio=('preco_medio', 'mean'),
    desconto_medio=('desconto_pct', 'mean'),
    taxa_promo=('flag_desconto_direto', 'mean'),
    taxa_ruptura=(
        'qt_ruptura',
        lambda s: s.gt(0).where(s.notna()).mean(),
    ),
).round(2)
print('\n--- ESTATÍSTICAS POR CANAL ---')
print(canal_stats)

# Comparação visual
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, variable, title, color in zip(
    axes,
    ['demanda_real', 'desconto_pct', 'preco_medio'],
    ['Demanda semanal (un.)', 'Desconto (%)', 'Preço médio (R$)'],
    ['#3f7180', '#d5a64a', '#7da3a0'],
):
    sns.boxplot(data=df, x='canal', y=variable, ax=ax, color=color, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_show(fig, '17_canal_comparacao')

"""
COMENTÁRIO: As estatística descritivas e a analise visual das distribuições de 
Demanda, Descono e Preço Médio mostram que existe uma politica de preço rigorosa entre canais visto que
a distribuição é quase idêntica. No período avaliado, a demanda semanal e levemente mais acentuada e com maior dispersão no canal Loja, 
bem como a aplicação de Desconto. Talvez isso indique uma tentativa de atrair o cliente para a experiência de Loja e não só como uma venda transacional 
OBS: Os pontos extremos foram ocultados para priorizar a distribuição central;
isso não significa que tenham sido removidos dos dados.
"""

#-------------------------------------------------------------------------------#
# A margem varia dentro do SKU ao longo do tempo, ou é estável?

# Variabilidade intra-SKU da margem
margem_stats = df.groupby('sku_id')['margem_pct'].agg(['mean','std','min','max'])
margem_stats['range'] = margem_stats['max'] - margem_stats['min']
print('\n--- VARIABILIDADE DA MARGEM POR SKU ---')
print(margem_stats.round(2).sort_values('range', ascending=False))

fig, ax = plt.subplots(figsize=(14, 5))
sns.boxplot(data=df, x='sku_id', y='margem_pct', ax=ax, color='#7da3a0', showfliers=False)
ax.set_title('A margem observada varia entre SKUs')
ax.set_xlabel(''); ax.set_ylabel('Margem (%)')
plt.xticks(rotation=45)
ax.spines[['top', 'right']].set_visible(False)
save_show(fig, '18_margem_por_sku')

"""
COMENTÁRIO: De forma geral, a variação da margem para os SKUs e relativamente pequena, 
o maior range foi SKU 04 com variação de 3,76 p.p, enquanto o SKU_02 com 1,4p.p foi o menor.
Entretando, quando avaliamos a margem entre SKUs, o SKU 04 trabalha com margens entre 26 e 30%, a menor entre todos os 
SKUs, bem distante da margem praticada pelo TOP SKU, SKU 01, e o SKU 11 com margens entre 57 e 61%.
"""