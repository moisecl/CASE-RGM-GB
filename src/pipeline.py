"""Executa a preparação dos dados e a modelagem."""

if __package__:
    from .data_utils import carregar_dados_vendas, preparar_dados_modelo
    from .model import run_model
else:
    from data_utils import carregar_dados_vendas, preparar_dados_modelo
    from model import run_model


def main() -> None:
    df = preparar_dados_modelo(carregar_dados_vendas())
    print(
        df.groupby("canal")[
            ["demanda_real", "preco_medio", "desconto_pct"]
        ].mean()
    )
    run_model(df)


if __name__ == "__main__":
    main()
