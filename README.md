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
A EDA gera os gráficos em `outputs/eda/`.
O modelo exclui os registros anteriores a 2025-01-01 antes de criar os atributos
temporais, pois 2024 tem apenas onze semanas observadas e possui comportamento atípico de demanda por SKU quando comparado à demanda de 2025 a 2026.

## Modelo e avaliação temporal

O alvo da regressão é `qt_vendida` (quantidade efetivamente vendida), não a
demanda estimada somando vendas e ruptura. Os lags e as médias móveis também são
calculados sobre `qt_vendida`, por SKU e canal, sempre com deslocamento de uma
semana para não usar a observação que será prevista.

São comparados LightGBM e XGBoost. Categorias de SKU, canal e subcategoria são
codificadas em one-hot. `preparar_dados_modelo`
faz os tratamentos causais em `src/data_utils.py`: preço, desconto e margem
ausentes usam histórico disponível, sem recorrer a semanas futuras ou à receita
da semana prevista. Ausências de ruptura e dos lags derivados são mantidas como
NaN e tratadas nativamente pelos estimadores. O LightGBM usa `linear_tree=True`
para permitir componentes lineares nas folhas ao usar `tendencia`; os demais
recursos representam histórico de venda, sazonalidade, ruptura anterior e
preço/promoção. Preço e mecânicas promocionais precisam ser conhecidos antes da
previsão (por exemplo, via plano comercial).

O corte é móvel e se baseia nas semanas presentes no CSV. As 8 semanas finais
ficam reservadas para teste. As 16 semanas imediatamente anteriores formam a
janela de validação cruzada, dividida em quatro janelas expansivas e contíguas
de 4 semanas; todas as semanas dessa janela recebem previsões fora da amostra
(OOF). Em cada fold, o treino contém todas as semanas anteriores à janela de
validação, inclusive o histórico anterior ao recorte de validação. O teste não
participa da seleção nem do ajuste.

Primeiro, os modelos padrão são comparados pela WAPE OOF nas 16 semanas (MAE
como desempate). O vencedor recebe uma busca aleatória simplificada de 12
combinações de parâmetros próprios daquele modelo. A versão ajustada é comparada
ao vencedor padrão nos mesmos folds; mantemos a que apresentar menor WAPE.
O modelo escolhido é ajustado em todas as semanas anteriores ao teste e avaliado
no teste final.
O teste tem duração de oito semanas como janela de avaliação — dois ciclos
mensais de planejamento S&OP, conforme a cadência descrita por Wallace e Stahl
em *Sales & Operations Planning: The How-To Handbook*, 3ª ed. O horizonte
operacional do estimador é, porém, uma semana à frente: a avaliação é
walk-forward e usa os dados da semana anterior, não uma previsão única de oito
semanas sem atualização. Como o CSV não informa lead times de fornecedores, a
janela bimestral é uma escolha de avaliação, não uma afirmação sobre o ciclo real
de compra; deve ser validada com a área de negócio.

Os folds respeitam a ordem do tempo: em nenhum deles o treino inclui semanas
posteriores às semanas de validação. Lags e médias móveis usam somente vendas
anteriores; os preenchimentos de preço, desconto, margem e subcategoria são
causais. Assim, não há vazamento temporal direto dessas variáveis. A busca de
parâmetros e a escolha do modelo reutilizam os mesmos folds, portanto a métrica
da validação pode ficar otimista por seleção repetida. Isso não expõe o target
do teste: o teste permanece fora da busca e da escolha e serve como avaliação
final independente.

Para facilitar a leitura do resultado no negócio, a execução salva dois
gráficos PNG em `outputs/model/`: `real_vs_previsto_venda_direta.png` e
`real_vs_previsto_loja.png`. Cada gráfico mostra as séries de venda real e
prevista no teste para os 20 SKUs daquele canal (40 combinações SKU-canal no
total). A previsão pontual é avaliada com WAPE, MAE em unidades e viés.

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
