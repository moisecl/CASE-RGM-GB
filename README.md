# Case RGM

## Estrutura

- `data/vendas_sku_semanal.csv`: dados de entrada.
- `src/data_utils.py`: caminhos, carregamento e preparação compartilhada.
- `src/eda.py`: análise exploratória; salva gráficos em `outputs/eda/`.
- `src/model.py`: treinamento, avaliação e interpretação do modelo.

Os caminhos de dados e de saída são resolvidos a partir da localização dos
módulos. Assim, os scripts podem ser executados sem depender do diretório atual.

## Deduplicação

Linhas idênticas são removidas. Quando uma chave `(sku_id, canal, semana)` tem
registros diferentes, é mantido um único registro original: primeiro se prioriza
o mais completo; em caso de empate, escolhe-se o mais próximo das medianas dos
valores numéricos daquela chave, com desempate pela ordem no arquivo. Isso evita
criar uma observação sintética por agregação e reduz a influência de valores
extremos. Como o arquivo não fornece uma coluna de atualização ou confiabilidade
da fonte, a ordem original é usada apenas como desempate determinístico.

## Preparação dos dados

Na EDA, as distribuições e os registros negativos, os descontos acima de 100%
e os preços acima de cinco vezes a mediana do SKU são exibidos antes da aplicação
das regras de limpeza. A função `preparar_valores_ausentes_vendas` tem dois
modos explícitos: a EDA usa `causal_imputation=False`, pois é análise descritiva
e pode usar a série completa; a preparação do modelo usa `True`, pois não pode
usar informação de semanas futuras.

Quantidade e receita são recuperadas quando os demais campos permitem usar a
relação `receita = quantidade × preço`; somente depois são removidas as linhas
que ainda não têm target ou receita. Na EDA, preço ausente/outlier pode ser
recalculado pela receita/quantidade e o restante preenchido pela mediana do SKU.
Se ainda houver preço ausente após esse fallback, a preparação da EDA gera erro
explícito, em vez de retornar uma tabela aparentemente completa. No modelo, o
preço usa somente histórico anterior; sem histórico, o NaN é mantido para
LightGBM/XGBoost, que tratam valores ausentes nativamente.

Subcategoria é imputada com mapeamento único de toda a série na EDA e apenas
com um mapeamento já observado no passado para o modelo. Margem e desconto usam
medianas de SKU-canal entre registros com receita positiva; no modelo, somente
valores anteriores são considerados. Descontos acima de 100% são inválidos.
Valores negativos de ruptura são limitados a zero, mas ausências de ruptura são
preservadas. Na detecção causal de outlier de preço, a primeira observação de
cada SKU-canal não é classificada: sem histórico anterior, o limite é NaN e a
comparação não marca o preço como outlier.

As features `preco_rel` e `demanda_real` não são criadas na preparação do modelo:
a primeira não é usada pelo estimador; a segunda é uma métrica exploratória e
seu filtro anterior dependia da mediana da série inteira, inclusive do teste.

## Instalação

```powershell
python -m pip install -r requirements.txt
```

## Execução

Na pasta `CASE_RGM_GB`:

```powershell
python src\eda.py
python src\model.py
```

`model.py` carrega e prepara os dados antes de treinar e avaliar os modelos.
A EDA gera os gráficos em `outputs/eda/` e a tabela
`14_elasticidade_promocional_por_sku.csv`. Ela estima, por SKU, a associação
entre preço e quantidade vendida em escala log-log, controlando mecânicas
promocionais, semana ISO do ano e canal. O intervalo de confiança de 95% da
elasticidade-preço usa bootstrap por semana. A análise considera somente
semanas sem ruptura, com venda positiva e preço observado plausível; SKUs com
menos de 20 graus de liberdade residuais são sinalizados como amostra
insuficiente, em vez de apresentar uma estimativa instável.
Por exemplo, elasticidade `-0,5` indica que, na amostra analisada, uma alta de
1% no preço está associada a uma queda aproximada de 0,5% na quantidade vendida.

Esses coeficientes são associações observacionais, não efeitos causais de uma
promoção. Os efeitos das flags comparam cada mecânica com a ausência daquela
flag, mantendo os demais fatores do modelo constantes; não estimam o efeito de
cada ponto percentual adicional de desconto nem um limite ótimo de desconto.


## Modelo e avaliação temporal

O alvo da regressão é `qt_vendida` (quantidade efetivamente vendida), não a
demanda estimada somando vendas e ruptura. Os lags e as médias móveis também são
calculados sobre `qt_vendida`, por SKU e canal, sempre com deslocamento de uma
semana para não usar a observação que será prevista.

O modelo exclui os registros anteriores a 2025-01-01 antes de criar os atributos
temporais, pois 2024 tem apenas onze semanas observadas e possui comportamento atípico de demanda por SKU quando comparado à demanda de 2025 a 2026.

São comparados LightGBM e XGBoost. Categorias de SKU, canal e subcategoria são
codificadas em one-hot. `preparar_dados_modelo`
faz os tratamentos causais em `src/data_utils.py`: preço, desconto e margem
ausentes usam histórico disponível, sem recorrer a semanas futuras ou à receita
da semana prevista. Para a previsão recursiva de oito semanas, as entradas do
modelo são SKU, canal, subcategoria, sazonalidade (`semana_seno` e
`semana_cosseno`) e histórico de venda (`venda_defasada_*`, `media_movel_*` e
`desvio_padrao_movel_4`). A cada semana do horizonte, lags e médias são
recalculados com as previsões anteriores do próprio modelo. Preço, promoção e
ruptura não entram no modelo porque o CSV não fornece seus valores planejados
para as semanas futuras; incluí-los com os valores realizados criaria
informação indisponível na data da previsão.

O corte é móvel e usa as semanas presentes no CSV. As oito semanas finais ficam
reservadas para o teste final. Antes delas, 32 semanas são usadas em validação
cruzada com quatro origens de previsão: cada bloco tem oito semanas, e o treino
de cada rodada contém todas as semanas anteriores ao respectivo bloco. Assim,
o treino cresce de uma rodada para a seguinte.

Em cada bloco, o modelo prevê as oito semanas em sequência. A previsão da
primeira semana é usada como histórico para prever a segunda, e assim por
diante; as vendas reais do bloco só são usadas depois para medir os erros. Os
modelos padrão são comparados pela WAPE agregada das quatro simulações (MAE
como desempate). A busca aleatória avalia 12 combinações de parâmetros pela
mesma simulação recursiva. O modelo selecionado é então ajustado com todo o
histórico anterior ao teste e gera uma única previsão recursiva para as oito
semanas finais, sem atualização com vendas reais. O teste não participa da
seleção nem do ajuste.

Como o CSV não informa lead times de fornecedores, a janela de oito semanas é
um horizonte de avaliação, não uma afirmação de que esse seja o prazo real de
compra; deve ser validada com a área de negócio. Os folds respeitam a ordem do
tempo e simulam o que estaria disponível na data de cada previsão. A busca de
parâmetros e a escolha do modelo reutilizam as mesmas janelas, portanto a
métrica da validação pode ficar otimista por seleção repetida. O teste final
permanece separado e fornece a avaliação independente.

Preço, promoção e ruptura futura são tratados como desconhecidos e não são
usados como entradas. Se a operação tiver planos futuros confiáveis para essas
variáveis, elas poderão ser adicionadas depois e avaliadas sem usar seus valores
realizados como substitutos.

Como o alvo é venda observada, e não demanda irrestrita, a previsão não corrige
por uma eventual ruptura futura que ainda não tenha sido informada.

Para facilitar a leitura do resultado no negócio, a execução salva dois
gráficos PNG em `outputs/model/`: `real_vs_previsto_venda_direta.png` e
`real_vs_previsto_loja.png`. Cada gráfico mostra as séries de venda real e
prevista no teste para os 20 SKUs daquele canal (40 combinações SKU-canal no
total), com uma faixa empírica P90. As previsões incluem o número da semana do
horizonte (`horizonte`), de 1 a 8.

Para mostrar quanto as previsões podem variar, usamos os erros das simulações
recursivas de validação. O erro P90 é calculado separadamente pela posição no
horizonte (semana 1 a 8) e pelo porte da previsão (menos de 50, de 50 a 150 e
mais de 150 unidades). Assim, uma previsão distante não herda automaticamente
o mesmo raio de erro de uma previsão próxima.

`erro_p90_unidades` é um erro de referência em unidades para cada combinação de
semana do horizonte e faixa de volume. P90 significa que, nas simulações usadas
para esse cálculo, 90% dos erros absolutos ficaram nesse valor ou abaixo dele.
Por exemplo, se uma previsão for de 100 unidades e o erro P90 daquela semana
do horizonte e faixa for 20, mostramos uma faixa de 80 a 120 unidades
(`previsao_inferior` e `previsao_superior`). A previsão inferior nunca fica
abaixo de zero.

`largura_faixa_pct` mostra o tamanho total da faixa comparado com a previsão. No
exemplo, a faixa vai de 80 a 120: sua largura é 40 unidades, ou 40% da previsão
de 100. Em geral, um percentual menor significa uma faixa mais estreita em
relação ao volume previsto. Para uma previsão igual a zero, esse percentual não
é calculado.

A tabela `erro_oof_por_horizonte_e_faixa`, retornada com os resultados, mostra
o erro P90 em unidades e a quantidade de observações para cada combinação de
semana do horizonte e porte da previsão. Se uma combinação não tiver
observações, o erro fica indefinido; não usamos o erro de outra faixa no lugar.
Essa faixa é uma referência baseada no histórico, não uma garantia de que o
valor real ficará dentro dela. O resultado também informa WAPE, MAE e viés por
semana do horizonte, tanto na validação quanto no teste.

O console também mostra SKU_01 como exemplo por canal: WAPE e erro absoluto
médio em unidades por semana, além de uma estimativa do valor bruto de vendas
associado ao erro, calculada como a média semanal de
`|real - previsto| × preco_medio`. Essa conversão é uma proxy para contextualizar
a escala do erro, não uma estimativa de prejuízo, custo de estoque ou custo de
ruptura: esses impactos dependem de margem, custos, nível de serviço, lead time
e decisões de reposição, informações que o CSV não fornece. Se não houver preço
válido no teste, o valor em reais não é estimado. Como o alvo é venda observada,
esse exemplo também não recupera a demanda que deixou de ser atendida durante
rupturas.

`best_hyperparameters` contém os parâmetros do estimador selecionado, e não de
uma busca candidata que tenha perdido.
