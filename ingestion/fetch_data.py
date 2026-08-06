import yfinance as yf

fiis = [
    "HGLG11.SA",
    "BTLG11.SA",
    "XPLG11.SA",
    "LVBI11.SA",
    "VILG11.SA",
]

for fii in fiis:
    dados = yf.download(
        fii,
        start="2010-01-01",
        auto_adjust=False,
        progress=False
    )

    # Remove o MultiIndex criado pelo yfinance
    dados.columns = dados.columns.get_level_values(0)

    nome_arquivo = fii.replace(".SA", "")

    dados.to_csv(
        f"ingestion/data/{nome_arquivo}_cotacoes.csv"
    )

    print(f"{nome_arquivo}: arquivo salvo com {len(dados)} linhas")