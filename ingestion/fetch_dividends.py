import pandas as pd
import re
import json
import html
import requests


def get_dividends(ticker):
    url = f"https://statusinvest.com.br/fundos-imobiliarios/{ticker}"

    r = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30
    )

    match = re.search(
        r'id="results".*?value="([^"]+)"',
        r.text,
        re.DOTALL
    )

    if match is None:
        raise ValueError(
            f"Não foi possível encontrar dividendos para {ticker}"
        )

    dados = html.unescape(match.group(1))
    dividendos = json.loads(dados)

    df = pd.DataFrame(dividendos)

    df["ed"] = pd.to_datetime(df["ed"], dayfirst=True)

    return df


for ticker in ["hglg11", "btlg11", "xplg11", "lvbi11", "vilg11"]:
    df = get_dividends(ticker)

    arquivo = f"ingestion/data/{ticker.upper()}_dividendos.csv"

    df.to_csv(arquivo, index=False)

    print(f"Salvo: {arquivo}")