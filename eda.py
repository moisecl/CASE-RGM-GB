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
        deduplicate_sales_data,
        load_sales_data,
    )
else:
    from data_utils import (
        EDA_OUTPUT_DIR,
        NUMERIC_COLUMNS,
        deduplicate_sales_data,
        load_sales_data,
    )

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
sns.set_theme(style='whitegrid', palette='viridis', context = 'talk')

FIGDIR = EDA_OUTPUT_DIR

def save_show(fig, name):
    """Salva toda figura."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f'{name}.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close(fig)

#-------------------------------------------------------------------------------#
# Qual é o shape, os tipos, o intervalo temporal e a granularidade real?
df = load_sales_data()

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
save_show(fig, '01_missing_heatmap')
plt.show()

"""
COMENTÁRIOS
- A valores ausentes em 7 colunas, concentrandos em margem_pct
- Pensando na construção do modelo preditivo, as 
colunas qt_ruptura, qt_vendida, receita são essenciais. Algum tratamento será necessário.
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

    print('\n--- EXEMPLOS DE DUPLICATAS QUASE (mostrar 5) ---')
    chaves_quase = variacao[variacao['tipo'].str.startswith('QUASE')][chave].head(5)
    print(df.merge(chaves_quase, on=chave, how='inner')
            .sort_values(chave)
            [chave + ['qt_vendida','qt_ruptura','receita','preco_medio','desconto_pct']]
            .round(2))
"""
Comentário: Após a análise, existem duplicatas exatas e "quase" (valores diferentes).
As duplicatas exatas podem ser removidas sem perda de informação.
As duplicatas "quase" foram tratatadas priorizando a completude e proximidade às medianas, 
preservando a ordem original em caso de empates.
Essa alternativa foi escolhida visando uma solução escalável e automatizada, 
sem necessidade de intervenção manual para cada caso.
Entretanto, no contexto desse case, poderia optar-se por uma solução mais simples 
como remover todas as duplicatas "quase" e manter apenas uma linha por chave sem nenhum critério adicional
"""

## Após a análise, remover exatas; para as "quase", manter uma linha observada por chave.
## Prioriza completude e proximidade às medianas; empates preservam a ordem original.
## Mais detalhes em data_utils.deduplicate_sales_data()
df = deduplicate_sales_data(df, chave)
num_cols = NUMERIC_COLUMNS
print(f'\nApós deduplicação, shape: {df.shape}')

#-------------------------------------------------------------------------------#
# Existem valores fisicamente impossíveis ou improváveis? Onde estão?

## Negativos: contagem por coluna
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
a fonte de dados não aparenta incluir valores de "devolução"(apenas 3 registros). Além disso,
a coluna receita está positiva e é igual ao qt_vendida(positiva) multiplicada pelo preço médio. 
Nesse contexto, optou-se por transformar a quantidade como qt_vendida em valores absolutos(positivos), para manter a coerência com a receita e o preço médio.
"""

df['qt_vendida_raw'] = df['qt_vendida']
df['qt_vendida'] = df['qt_vendida'].abs()

print('\n--- DESCONTO > 100%) ---')
print(df[df['desconto_pct'] > 100]
        [['sku_id','canal','semana_dt','qt_vendida','desconto_pct','preco_medio','margem_pct','receita']]
        .sort_values('desconto_pct', ascending=False).head(10))
"""
COMENTÁRIO: Transformar deconto_pct com base nos valores aproximados de preçoe médio e margem_pct
"""

# Distribuição de preço médio POR SKU (boxplot) — evidencia outliers visuais
fig, ax = plt.subplots(figsize=(16, 5))
sns.boxplot(data=df, x='sku_id', y='preco_medio', ax=ax, color='steelblue')
ax.set_title('Distribuição de preço médio por SKU (cada caixa = 1 SKU)')
ax.set_xlabel(''); ax.set_ylabel('Preço médio (R$)')
plt.xticks(rotation=45)
save_show(fig, '02_preco_por_sku_boxplot')

# Regra: marcar como NaN quando preço > 5× mediana do PRÓPRIO SKU.
# Escolhi 5× (não 3×) para não pegar picos legítimos de preço premium.
med_sku = df.groupby('sku_id')['preco_medio'].transform('median')
outliers_preco = df[df['preco_medio'] > 5 * med_sku][['sku_id','semana_dt','preco_medio']]
print('\n--- OUTLIERS DE PREÇO (> 5× mediana do próprio SKU) ---')
print(outliers_preco.sort_values('preco_medio', ascending=False))

# Aplicar
df.loc[df['preco_medio'] > 5 * med_sku, 'preco_medio'] = np.nan
df['preco_medio'] = df.groupby('sku_id')['preco_medio'].transform(
    lambda s: s.fillna(s.median()))
#-------------------------------------------------------------------------------#

# Ruptura — frequência, padrão temporal, relação com a venda
 ## Perguntas:
 ### A ruptura é episódica ou sistemática?
 ### Quando ela ocorre (sazonalidade)?
 ### Ela reduz a venda observada?

## Estatísticas de ruptura por SKU
rupt_stats = (
    df.assign(teve_ruptura=(df['qt_ruptura'] > 0))
      .groupby('sku_id')
      .agg(semanas=('qt_ruptura', 'size'),
           taxa_ruptura=('teve_ruptura', 'mean'),
           ruptura_media=('qt_ruptura', 'mean'),
           ruptura_max=('qt_ruptura', 'max'))
      .sort_values('taxa_ruptura', ascending=False)
)
print('--- TAXA DE RUPTURA POR SKU ---')
print(rupt_stats.round(2))

## Visualização: taxa por SKU
fig, ax = plt.subplots(figsize=(14, 5))
sns.barplot(x=rupt_stats.index, y='taxa_ruptura', data=rupt_stats.reset_index(), ax=ax)
ax.axhline(rupt_stats['taxa_ruptura'].median(), color='red', linestyle='--',
           label=f'Mediana = {rupt_stats["taxa_ruptura"].median():.1%}')
ax.set_title('Taxa de semanas com ruptura por SKU (todos os canais)')
ax.set_ylabel('Fração das semanas com qt_ruptura > 0')
ax.set_xlabel(''); ax.tick_params(axis='x', rotation=45)
ax.legend()
save_show(fig, '03_taxa_ruptura_por_sku')

## Comparação: distribuição de qt_vendida nas semanas COM vs SEM ruptura
fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(data=df.assign(ruptura_grupo=np.where(df['qt_ruptura']>0,
                                                      'com ruptura', 'sem ruptura')),
               x='ruptura_grupo', y='qt_vendida', ax=ax, split=False, inner='quart')
ax.set_title('Distribuição de qt_vendida: semanas com vs. sem ruptura')
ax.set_ylabel('Quantidade vendida')
save_show(fig, '04_violin_ruptura_vs_venda')

# Comparação numérica
comp = df.assign(com_ruptura=(df['qt_ruptura'] > 0)).groupby('com_ruptura')['qt_vendida'].agg(['mean','median','count'])
print('\n--- MÉDIA/MEDIANA DE qt_vendida: COM vs SEM ruptura ---')
print(comp.round(2))


## Criar coluna de demanda real (qt_vendida + qt_ruptura)
df['demanda_real'] = df['qt_vendida'] + df['qt_ruptura'].fillna(0)

#-------------------------------------------------------------------------------#

# A demanda tem padrão de semana do ano? É uniforme entre SKUs/canais?
df['semana_iso'] = df['semana_dt'].dt.isocalendar().week.astype(int)
df['mes'] = df['semana_dt'].dt.month
df['trimestre'] = df['semana_dt'].dt.quarter

# Sazonalidade agregada — média por semana ISO
saz_geral = df.groupby('semana_iso')['demanda_real'].mean()
fig, ax = plt.subplots(figsize=(16, 5))
sns.lineplot(x=saz_geral.index, y=saz_geral.values, ax=ax, marker='o', color='darkblue')
ax.axvspan(48, 52, alpha=0.15, color='red', label='Semanas 48–52 (Natal/Ano Novo)')
ax.axvspan(6, 13, alpha=0.15, color='orange', label='Semanas 6–13 (pós-festas)')
ax.set_title('Demanda média por semana ISO — todos os SKUs e canais')
ax.set_xlabel('Semana do ano'); ax.set_ylabel('Demanda média (un.)')
ax.legend()
save_show(fig, '05_sazonalidade_semana_iso')

# Sazonalidade por mês
fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df, x='mes', y='demanda_real', ax=ax)
ax.set_title('Distribuição de demanda por mês (todos os SKUs e canais)')
ax.set_xlabel('Mês'); ax.set_ylabel('Demanda semanal (un.)')
save_show(fig, '06_sazonalidade_mes')

# Sazonalidade por categoria (split de canais)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
for ax, cat in zip(axes, sorted(df['categoria_id'].unique())):
    d = df[df['categoria_id'] == cat].groupby('semana_iso')['demanda_real'].mean().reset_index()
    sns.lineplot(data=d, x='semana_iso', y='demanda_real', ax=ax, color='teal')
    ax.set_title(f'Categoria {cat}')
    ax.set_xlabel('Semana ISO'); ax.set_ylabel('Demanda média')
plt.tight_layout()
save_show(fig, '07_sazonalidade_por_categoria')

#-------------------------------------------------------------------------------#

# Os SKUs têm tendência, nível e volatilidade semelhantes? Quais são "anormais"?

## Small multiples: uma linha por SKU (Loja)
loja = df[df['canal'] == 'Loja'].sort_values(['sku_id','semana_dt']).copy()
skus = sorted(loja['sku_id'].unique())

fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 12), sharex=False)
for ax, sku in zip(axes.flat, skus):
    d = loja[loja['sku_id'] == sku]
    sns.lineplot(data=d, x='semana_dt', y='demanda_real',
                 ax=ax, color='steelblue', linewidth=1.2)
    ax.set_title(sku, fontsize=10)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=7, rotation=30)
plt.suptitle('Demanda semanal por SKU — canal Loja (eixo y varia por SKU)', y=1.02)
plt.tight_layout()
save_show(fig, '08_ts_por_sku_loja')

#-------------------------------------------------------------------------------#
# Os picos de demanda se concentram nas mesmas semanas para todos os SKUs, ou cada um tem seu calendário?

pivot = loja.pivot_table(index='sku_id', columns='semana_iso',
                         values='demanda_real', aggfunc='mean')
fig, ax = plt.subplots(figsize=(22, 8))
sns.heatmap(pivot, cmap='YlOrRd', cbar_kws={'label': 'Demanda média (un.)'}, ax=ax)
ax.set_title('Demanda média por SKU × semana do ano — Loja')
ax.set_xlabel('Semana ISO'); ax.set_ylabel('')
save_show(fig, '09_heatmap_sku_semana')

#-------------------------------------------------------------------------------#
# As mecânicas são usadas com frequência semelhante?
# Qual mecânica gera mais lift em cada SKU?

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
            x='mecanica', y='desconto_pct', ax=ax)
ax.set_title('Profundidade de desconto por mecânica (só semanas com promoção)')
ax.set_xlabel(''); ax.set_ylabel('Desconto (%)')
save_show(fig, '10_desconto_por_mecanica')

# Boxplot de demanda por mecânica — pequenos múltiplos por SKU (top 8 volume)
top_skus = df.groupby('sku_id')['demanda_real'].mean().nlargest(8).index
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, sku in zip(axes.flat, top_skus):
    d = df[df['sku_id'] == sku]
    sns.boxplot(data=d, x='mecanica', y='demanda_real', ax=ax)
    ax.set_title(sku, fontsize=10)
    ax.set_xlabel(''); ax.set_ylabel('Demanda (un.)')
    ax.tick_params(axis='x', labelsize=8, rotation=20)
plt.suptitle('Demanda por mecânica — top 8 SKUs por volume', y=1.02)
plt.tight_layout()
save_show(fig, '11_boxplot_demanda_por_mecanica')


#-------------------------------------------------------------------------------#
# Qual é a forma da relação preço–demanda por SKU? Ela é compatível com uma elasticidade estável?

# Scatter log-log por SKU, restrito a semanas SEM ruptura
loja_ok = loja[(loja['qt_ruptura'] == 0) & (loja['qt_vendida'] > 0) & (loja['preco_medio'] > 0)].copy()
fig, axes = plt.subplots(4, 5, figsize=(20, 12), sharex=False)
for ax, sku in zip(axes.flat, skus):
    d = loja_ok[loja_ok['sku_id'] == sku].copy()
    if len(d) < 5:
        ax.text(0.5, 0.5, 'poucos pontos', ha='center', va='center')
        ax.set_title(sku); continue
    d['ln_q'] = np.log(d['qt_vendida'])
    d['ln_p'] = np.log(d['preco_medio'])
    sns.regplot(data=d, x='ln_p', y='ln_q', ax=ax, scatter_kws={'alpha': 0.5, 's': 15},
                line_kws={'color': 'red', 'linewidth': 1.5})
    ax.set_title(sku, fontsize=10)
    ax.set_xlabel(''); ax.set_ylabel('')
plt.suptitle('log(demanda) vs. log(preço) — Loja, semanas sem ruptura (linha = OLS simples)', y=1.02)
plt.tight_layout()
save_show(fig, '12_loglog_elasticidade')

## Estimativa com controle (mecânica + mês) — para evitar viés de variável omitida
import statsmodels.formula.api as smf

def elast_sku(d):
    d = d[(d['qt_ruptura'] == 0) & (d['qt_vendida'] > 0) & (d['preco_medio'] > 0)].copy()
    if len(d) < 20:
        return pd.Series({'elast': np.nan, 'pval': np.nan, 'n': len(d)})
    d['ln_q'] = np.log(d['qt_vendida']); d['ln_p'] = np.log(d['preco_medio'])
    try:
        m = smf.ols('ln_q ~ ln_p + flag_desconto_direto + flag_combo + flag_pague_e_leve + C(mes)',
                    data=d).fit()
        return pd.Series({'elast': m.params['ln_p'], 'pval': m.pvalues['ln_p'], 'n': len(d)})
    except Exception:
        return pd.Series({'elast': np.nan, 'pval': np.nan, 'n': len(d)})

elast = df.groupby(['sku_id','canal']).apply(elast_sku).reset_index().sort_values('elast')
print('\n--- ELASTICIDADE POR SKU × CANAL (controle: mecânica + mês; só semanas sem ruptura) ---')
print(elast.round(3))

## Visual: elasticidade e significância
fig, ax = plt.subplots(figsize=(14, 6))
elast_plot = elast.dropna(subset=['elast']).copy()
elast_plot['sig'] = np.where(elast_plot['pval'] < 0.05, 'significativa (p<0,05)', 'não significativa')
sns.barplot(data=elast_plot.sort_values('elast'), x='elast', y='sku_id', hue='canal',
            ax=ax, orient='h')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Elasticidade estimada por SKU × canal')
ax.set_xlabel('Elasticidade (ln q / ln p)')
save_show(fig, '13_elasticidade_sku_canal')

#-------------------------------------------------------------------------------#
# SKUs da mesma subcategoria têm demanda correlacionada? Promoção de um afeta o outro?

# Correlação de demanda entre SKUs da mesma subcategoria
for subcat, g in df.groupby('subcategoria_id'):
    if g['sku_id'].nunique() < 2: continue
    pivot = (g[g['canal'] == 'Loja']
               .pivot_table(index='semana_dt', columns='sku_id',
                            values='demanda_real', aggfunc='mean'))
    if pivot.shape[0] < 20 or pivot.shape[1] < 2: continue
    corr = pivot.corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax)
    ax.set_title(f'Correlação de demanda — {subcat} (Loja)')
    save_show(fig, f'14_corr_{subcat}')

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
    taxa_ruptura=('qt_ruptura', lambda s: (s > 0).mean()),
).round(2)
print('\n--- ESTATÍSTICAS POR CANAL ---')
print(canal_stats)

# Comparação visual
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(data=df, x='canal', y='demanda_real', ax=axes[0]); axes[0].set_title('Demanda semanal')
sns.boxplot(data=df, x='canal', y='desconto_pct', ax=axes[1]); axes[1].set_title('Desconto (%)')
sns.boxplot(data=df, x='canal', y='preco_medio', ax=axes[2]); axes[2].set_title('Preço médio (R$)')
plt.tight_layout()
save_show(fig, '15_canal_comparacao')

#-------------------------------------------------------------------------------#
# A margem varia dentro do SKU ao longo do tempo, ou é estável?

# Variabilidade intra-SKU da margem
margem_stats = df.groupby('sku_id')['margem_pct'].agg(['mean','std','min','max'])
margem_stats['range'] = margem_stats['max'] - margem_stats['min']
print('\n--- VARIABILIDADE DA MARGEM POR SKU ---')
print(margem_stats.round(2).sort_values('range', ascending=False))

fig, ax = plt.subplots(figsize=(14, 5))
sns.boxplot(data=df, x='sku_id', y='margem_pct', ax=ax)
ax.set_title('Distribuição de margem (%) por SKU — quão estável é dentro de cada SKU?')
ax.set_xlabel(''); ax.set_ylabel('Margem (%)')
plt.xticks(rotation=45)
save_show(fig, '16_margem_por_sku')

#-------------------------------------------------------------------------------#
# Existem semanas com demanda_real = 0 em SKUs de alto volume? Isso é plausível?

# Identificar zeros em SKUs com mediana alta
med_sku = df.groupby('sku_id')['demanda_real'].transform('median')
zeros_suspeitos = df[(df['demanda_real'] == 0) & (med_sku > 20)].copy()
print(f'\nZeros "suspeitos" (SKU com mediana > 20 un./sem e demanda = 0): {len(zeros_suspeitos)}')
print(zeros_suspeitos[['sku_id','canal','semana_dt','demanda_real','qt_ruptura','receita']]
        .sort_values(['sku_id','semana_dt']).head(20))

# Visual: marcar zero na série do SKU_01 (alto volume)
d = loja[loja['sku_id'] == 'SKU_01']
fig, ax = plt.subplots(figsize=(16, 4))
sns.lineplot(data=d, x='semana_dt', y='demanda_real', ax=ax, color='steelblue')
zeros = d[d['demanda_real'] == 0]
ax.scatter(zeros['semana_dt'], zeros['demanda_real'], color='red', s=60,
           zorder=5, label='Demanda = 0 (suspeita)')
ax.set_title('SKU_01 — Loja: semanas com demanda registrada como 0')
ax.legend()
save_show(fig, '17_zeros_suspeitos_sku01')

df['zero_suspeito'] = (df['demanda_real'] == 0) & (med_sku > 20)
df.loc[df['zero_suspeito'], 'demanda_real'] = np.nan
print(f'\nObservações marcadas como NaN no alvo: {df["zero_suspeito"].sum()}')