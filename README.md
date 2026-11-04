# Case RGM

## Estrutura

- `data/vendas_sku_semanal.csv`: dados de entrada.
- `src/data_utils.py`: caminhos, carregamento e preparação compartilhada.
- `src/eda.py`: análise exploratória; salva gráficos em `outputs/eda/`.
- `src/model.py`: treinamento, avaliação e interpretação do modelo.
- `src/pipeline.py`: executa preparação, resumo de elasticidade e modelagem.

Os caminhos de dados e de saída são resolvidos a partir da localização dos
módulos. Assim, os scripts podem ser executados sem depender do diretório atual.

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
