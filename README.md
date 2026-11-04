# Case RGM

## Estrutura

- `data/vendas_sku_semanal.csv`: dados de entrada.
- `src/data_utils.py`: caminhos, carregamento e preparação compartilhada.
- `src/eda.py`: análise exploratória; salva gráficos em `outputs/eda/`.
- `src/model.py`: treinamento, avaliação e interpretação do modelo.
- `src/pipeline.py`: executa preparação, resumo de elasticidade e modelagem.

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
das regras de limpeza. Depois do diagnóstico, a função
`preparar_valores_ausentes_vendas` é chamada para os dados da EDA e também pela
preparação do modelo.

Quantidade e receita são recalculadas somente quando os outros dois campos
permitem obter o valor pela relação `receita = quantidade × preço`. Preço é
recalculado quando quantidade positiva e receita estão disponíveis; demais
ausências e outliers são preenchidos pela mediana do SKU. Subcategoria é
preenchida apenas quando SKU, categoria e canal têm mapeamento observado único.
Margem e desconto são imputados pela mediana do SKU-canal, usando como referência
registros com receita positiva; descontos acima de 100% são tratados como
inválidos antes da imputação. Valores negativos de ruptura são limitados a zero,
mas ausências de ruptura e de desconto em registros sem receita positiva são
preservadas.

## Instalação

```powershell
python -m pip install -r requirements.txt
```

## Execução

Na pasta `CASE_RGM_GB`:

```powershell
python src\eda.py
python src\model.py
python src\pipeline.py
```

`pipeline.py` executa o resumo de elasticidade e o modelo. Use `model.py` para
treinar e avaliar somente o modelo. A EDA gera os gráficos em `outputs/eda/`.
O modelo exclui os registros anteriores a 2025-01-01 antes de criar os atributos
temporais, pois 2024 tem apenas onze semanas observadas e possui comportamento atípico de demanda por SKU quando comapardao a demanda de 2025 a 2026.
