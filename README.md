# Case RGM

## Visão geral

Este projeto analisa vendas semanais por SKU e canal e compara modelos para
prever a quantidade vendida nas oito semanas seguintes. A análise exploratória
investiga sazonalidade, rupturas, promoções, elasticidade-preço e relações entre
SKUs. A modelagem usa validação temporal, previsão recursiva e uma faixa
quantílica para comunicar a incerteza das previsões.

O alvo do modelo é `qt_vendida`, vendas observadas, e não a demanda estimada
somando vendas e ruptura. Como o arquivo não informa planos futuros de preço,
promoção ou ruptura, essas variáveis não são usadas como entradas da previsão.

## Estrutura do projeto

- `data/vendas_sku_semanal.csv`: dados de entrada, uma observação por SKU,
  canal e semana.
- `src/data_utils.py`: carregamento, deduplicação e preparação compartilhada.
- `src/eda.py`: análise exploratória, gráficos e estimativa de elasticidade.
- `src/model.py`: treinamento, avaliação, explicações SHAP locais e gráficos
  de previsão; este é o workflow recomendado para execução.
- `src/model_quantile.py`: script independente que também executa o workflow
  quantílico.
- `outputs/eda/`: gráficos e tabela de elasticidade gerados pela EDA.
- `outputs/model/`: gráficos de vendas reais e previstas e explicações SHAP.

Os caminhos de entrada e saída são resolvidos a partir da localização dos
módulos; não dependem do diretório atual.

## Requisitos e execução

Requer Python 3.11 ou superior. Instale as dependências fixadas no arquivo:

```powershell
python -m pip install -r requirements.txt
```

Na pasta do projeto, execute:

```powershell
python src\eda.py
python src\model.py
```

O CSV precisa estar em `data/vendas_sku_semanal.csv`. A EDA salva os gráficos
em `outputs/eda/` e a tabela
`14_elasticidade_promocional_por_sku.csv`; também exibe os gráficos durante a
execução. O modelo salva `real_vs_previsto_venda_direta.png` e
`real_vs_previsto_loja.png`, além de dois gráficos SHAP Waterfall em
`outputs/model/`. Também imprime métricas e exemplos de previsão no console.

## Análise exploratória

A EDA descreve a qualidade e a cobertura dos dados, além de explorar:

- ocorrência, volume e valor estimado das rupturas;
- sazonalidade por semana, mês, categoria, SKU e canal;
- distribuição de descontos e demanda por mecânica promocional;
- correlações de demanda entre SKUs da mesma subcategoria;
- variação de margem e diferenças entre canais;
- associação entre preço e quantidade vendida.

A elasticidade-preço é estimada por SKU em escala log-log, controlando
mecânicas promocionais, semana ISO e canal. O intervalo de confiança de 95% é
obtido por bootstrap de semanas. A análise usa semanas sem ruptura, com venda
positiva e preço observado plausível. SKUs com menos de 20 graus de liberdade
residuais são sinalizados como insuficientes para uma estimativa estável.

Por exemplo, uma elasticidade de `-0,5` indica que, na amostra, uma alta de 1%
no preço está associada a uma queda aproximada de 0,5% na quantidade vendida.
Os coeficientes e as comparações promocionais são associações observacionais,
não efeitos causais. Os efeitos das flags comparam cada mecânica com a ausência
daquela flag, mantendo os demais fatores constantes; não estimam o efeito de
cada ponto percentual de desconto nem um desconto ótimo.

## Modelo e avaliação

### Atributos e previsão

São comparados LightGBM e XGBoost. SKU, canal e subcategoria são codificados
em one-hot. Os atributos numéricos incluem sazonalidade anual
(`semana_seno`, `semana_cosseno`) e histórico de venda por SKU e canal:
defasagens de uma, duas, quatro e oito semanas, médias móveis de quatro e oito
semanas e desvio-padrão móvel de quatro semanas. As estatísticas históricas são
defasadas em uma semana para não incluir o resultado previsto.


Cada previsão cobre oito semanas recursivamente: as previsões anteriores do
próprio modelo são usadas para atualizar lags e médias móveis dentro do
horizonte. Preço, promoção e ruptura futura não são entradas, pois seus valores
planejados não estão disponíveis no CSV. Usar valores realizados para essas
semanas introduziria informação que não estaria disponível no momento da
previsão.

### Validação temporal

O corte usa as semanas presentes no CSV. As oito semanas finais são reservadas
para o teste. As 32 semanas anteriores formam quatro blocos de validação de
oito semanas; em cada origem, o treino contém somente semanas anteriores ao
bloco de validação. O treino cresce de uma rodada para a seguinte.

Em cada bloco, o modelo prevê as oito semanas em sequência sem usar as vendas
reais do bloco para atualizar o histórico. Os modelos padrão são comparados
pela WAPE agregada da validação, usando MAE como desempate. Depois, 12
combinações de hiperparâmetros do modelo padrão vencedor são avaliadas nas
mesmas janelas e pela mesma previsão recursiva. O modelo selecionado é ajustado
com todo o período anterior ao teste e gera uma única previsão recursiva para
as oito semanas finais. O teste não participa da seleção nem do ajuste.

A validação e a busca reutilizam as mesmas janelas; a seleção repetida pode
deixar as métricas de validação otimistas. O teste final permanece separado
para fornecer uma avaliação independente.

### Métricas e interpretação

O resultado apresenta WAPE, MAE e BIAS:

- **WAPE**: soma dos erros absolutos dividida pela soma das vendas reais
  absolutas. Resume o erro relativo agregado; menor é melhor.
- **MAE**: média dos erros absolutos, em unidades vendidas. Indica o tamanho
  típico do erro; menor é melhor.
- **BIAS**: erro médio (previsto menos real) dividido pela média real. Valor
  positivo indica tendência a superestimar; negativo, a subestimar; zero indica
  ausência de viés médio. Valores positivos e negativos podem se compensar.

As métricas são apresentadas para validação e teste; WAPE e MAE também são
detalhadas por semana do horizonte. O console inclui ainda um exemplo de
tradução do erro do SKU_01 por canal. A conversão para reais usa
`|real - previsto| × preco_medio` como proxy de valor bruto associado ao erro;
não é uma estimativa de prejuízo, custo de estoque ou custo de ruptura.

Depois que o melhor modelo-base é selecionado pela WAPE/MAE da validação, o
estimador escolhido (LightGBM ou XGBoost, com seus parâmetros selecionados) é
ajustado separadamente para os quantis 0,05, 0,50 e 0,95. Essa etapa não altera
a escolha do modelo-base. A previsão pontual `previsao` é P50; `previsao_p05` e
`previsao_p95` delimitam o intervalo quantílico exibido nos gráficos. Como os
quantis são ordenados após limitar valores negativos a zero, as três saídas
permanecem não negativas e sem cruzamento. P50 é realimentado no histórico para
as semanas seguintes da previsão recursiva.

O intervalo P05–P95 tem cobertura nominal de 90% (0,95 − 0,05); a cobertura
efetivamente observada pode diferir. A validação OOF e o teste reportam
*pinball loss* de cada quantil, cobertura observada por quantil, cobertura do
intervalo P05–P95 e largura média, também detalhadas por semana do horizonte.
Esses diagnósticos medem o desempenho nas amostras avaliadas; não são garantia
de cobertura futura nem intervalo de confiança.

- **Pinball loss**: função de erro assimétrica que avalia cada previsão
  quantílica; menor é melhor e cada quantil é avaliado separadamente.

No resultado de `run_model`, `quantile_models` contém os três pipelines finais,
`quantile_validation_predictions` contém as previsões OOF e `metrics` inclui
`quantile_cross_validation` e `quantile_test`. Em cada previsão,
`largura_faixa_pct` é a largura P95–P05 como percentual de P50; fica indefinida
se P50 for zero.
O retorno `model` também corresponde ao pipeline P50; `best_hyperparameters`
registra os parâmetros do modelo-base selecionado.

Para explicar previsões individuais, são salvos dois gráficos SHAP Waterfall:
um para SKU_01 na primeira semana do teste em cada canal
(`shap_waterfall_sku_01_venda_direta.png` e
`shap_waterfall_sku_01_loja.png`). O ponto de partida é a previsão-base do
modelo; cada barra mostra quanto uma variável aumenta ou reduz a saída até
chegar à previsão daquela linha. As categorias one-hot são agrupadas pelo
atributo original, e as contribuições são expressas em unidades previstas.
As features usadas são exatamente as da previsão recursiva, incluindo as
vendas previstas usadas como histórico quando aplicável.

Esses gráficos explicam exemplos específicos, não todas as previsões do teste.
SHAP descreve como o modelo chegou à saída para aquelas entradas; não demonstra
causalidade. Se a saída bruta do modelo for negativa, o gráfico explica essa
saída antes do limite mínimo aplicado para apresentar previsões não negativas.

## Preparação dos dados

Linhas exatamente duplicadas são removidas. Para registros diferentes com a
mesma chave `(sku_id, canal, semana)`, mantém-se uma linha original: prioriza-se
a de maior completude; em caso de empate, escolhe-se a mais próxima das
medianas dos valores numéricos da chave. A ordem no arquivo desempata os casos
restantes. Não se agrega o grupo em uma observação sintética, pois não há
coluna de atualização ou confiabilidade da fonte.

Na EDA, distribuições e registros potencialmente inválidos são examinados
antes da limpeza. A função `preparar_valores_ausentes_vendas` tem dois modos:
a EDA pode usar a série completa, enquanto a preparação do modelo usa somente
informações anteriores para imputar atributos.

Quantidade e receita são recuperadas quando os demais campos permitem usar
`receita = quantidade × preço`; linhas que continuam sem target ou receita
são removidas. Quantidades negativas são convertidas em valores absolutos.
Preços ausentes ou extremos podem ser reconstruídos pela receita dividida pela
quantidade; na EDA, os casos restantes usam a mediana do SKU. No modelo, o
preço usa apenas histórico anterior e continua ausente se não houver referência
histórica. Preços acima de cinco vezes a mediana de referência do SKU são
tratados como extremos; no modelo, a referência é calculada apenas com o
histórico anterior do SKU-canal.

Subcategoria é imputada com um mapeamento único da série na EDA e apenas com
um mapeamento já observado no passado para o modelo. Margem e desconto usam
medianas de SKU-canal entre registros com receita positiva; no modelo, somente
valores anteriores são considerados. Descontos acima de 100% são tratados
como inválidos. Rupturas negativas são limitadas a zero e valores ausentes de
ruptura são preservados. Na detecção causal de preços extremos, a primeira
observação de cada SKU-canal não é classificada por não haver histórico
anterior.

`preco_rel` não é usado pelo estimador. `demanda_real` é uma métrica exploratória
e não é o alvo do modelo, que prevê vendas observadas. Por isso, a previsão não
corrige por uma ruptura futura ainda não informada.

## Limitações e termos

- O CSV não informa lead time de fornecedores. O horizonte de oito semanas é
  uma escolha de avaliação, não uma afirmação sobre o prazo de compra; deve
  ser validado com a área de negócio.
- Sem planos futuros confiáveis de preço, promoção e ruptura, essas variáveis
  são consideradas desconhecidas. Se os planos passarem a existir, podem ser
  incorporados e avaliados sem substituí-los por valores realizados.
- A EDA pode usar estatísticas da série completa para fins descritivos; esse
  modo não é usado na preparação causal do modelo.
- **One-hot**: representação de uma categoria por colunas indicadoras.
- **Fold**: uma rodada de validação; neste projeto, cada bloco valida oito
  semanas e treina apenas com o passado.
- **OOF (out-of-fold)**: previsões feitas em blocos de validação que não foram
  usados para ajustar o modelo daquela rodada.
- **P05/P50/P95**: quantis condicionais 0,05, 0,50 (mediana) e 0,95 da
  distribuição prevista; P05–P95 tem cobertura nominal de 90%.
- **Bootstrap**: reamostragem de semanas para estimar a variabilidade da
  elasticidade.
- **SHAP Waterfall**: gráfico de explicação local que decompõe uma previsão em
  uma saída-base mais contribuições positivas e negativas das variáveis.
- **NaN**: valor ausente. Os modelos de árvore usados aqui tratam valores
  numéricos ausentes diretamente.
