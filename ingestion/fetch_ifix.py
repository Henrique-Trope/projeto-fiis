import pandas as pd

anos = range(2011, 2027)
dados_anuais = []

meses = {
    "Jan": 1,
    "Fev": 2,
    "Mar": 3,
    "Abr": 4,
    "Mai": 5,
    "Jun": 6,
    "Jul": 7,
    "Ago": 8,
    "Set": 9,
    "Out": 10,
    "Nov": 11,
    "Dez": 12
}

for ano in anos:
    caminho_arquivo = f"ingestion/data/IFIX_{ano}.csv"

    ifix = pd.read_csv(
        caminho_arquivo,
        sep=";",
        encoding="latin-1",
        dtype=str,
        skiprows=1
    )

    ifix = ifix[
        pd.to_numeric(ifix["Dia"], errors="coerce").notna()
    ]

    ifix = ifix.melt(
        id_vars="Dia",
        var_name="Mes",
        value_name="Fechamento"
    )

    ifix = ifix.dropna(subset=["Fechamento"])

    ifix["Fechamento"] = (
        ifix["Fechamento"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    ifix["Data"] = pd.to_datetime({
        "year": ano,
        "month": ifix["Mes"].map(meses),
        "day": ifix["Dia"].astype(int)
    })

    ifix = ifix[["Data", "Fechamento"]]

    dados_anuais.append(ifix)

    print(f"{ano}: {len(ifix)} registros processados")

ifix_completo = pd.concat(
    dados_anuais,
    ignore_index=True
)

ifix_completo = (
    ifix_completo
    .drop_duplicates(subset=["Data"])
    .sort_values("Data")
    .reset_index(drop=True)
)

print(ifix_completo.shape)
print(ifix_completo.head())
print(ifix_completo.tail())

ifix_completo.to_csv(
    "ingestion/data/IFIX_completo.csv",
    index=False
)