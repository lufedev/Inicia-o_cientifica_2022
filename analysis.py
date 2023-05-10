import pandas.io.sql as sqlio
import psycopg2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools as ft
import os
from ydata_profiling import ProfileReport
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.tree import export_graphviz
from graphviz import Source
from sklearn.metrics import accuracy_score
from six import StringIO
from IPython.display import Image
import pydotplus
from sklearn import (
    metrics,
)  # Import scikit-learn metrics module for accuracy calculation

# Load libraries

# COISAS IMPORTANTES
diretorio = "./dados/Filtered/"
conn = psycopg2.connect(
    # 192.168.15.45
    "dbname = 'postgres' user = 'postgres' host = '192.168.15.45' port = '7777' password = 'ic_2023'"
)
uf = [
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
]
anos = [
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
]
df_ano = {}
df_filtrado = {}
df_estado = {}
UF = {}


# # QUERIES
def getByYear(year):
    query = 'select * from "raw_data" WHERE ano=' + year + ";"
    return sqlio.read_sql_query(query, conn)


# def getAnalfabetismo(year):
#     query = 'select "UF","' + year + '" from "analfabetismo";'
#     return sqlio.read_sql_query(query, conn)


# def getRenda(year):
#     query = 'select "UF","' + year + '" from "rendapercapita";'
#     return sqlio.read_sql_query(query, conn)


def separarPorCritério(df, criterio, variable, filename):
    df_filtrado[filename] = df.loc[df[criterio] == variable]
    # df.loc[df[criterio] == variable].to_csv(
    #     "/home/luiz/Documentos/Iniciacao_cientifica_2022/dados/porUF/" + filename,
    #     index=False,
    # )


def genDecisionTree(X, y):
    print("")


# Coleta a cv anual
for ano in anos:
    df_ano[ano] = getByYear(ano)
    for estado in uf:
        separarPorCritério(df_ano[ano], "estado_abrev", estado, ano + "_" + estado)

for estado in uf:
    dfs_estado = []
    for ano in anos:
        df_ano_estado = df_filtrado[ano + "_" + estado]
        if not df_ano_estado.empty:
            dfs_estado.append(df_ano_estado)
    if len(dfs_estado) > 0:
        df_estado[estado] = pd.concat(dfs_estado)

feature_cols = ["exp_vida", "idhm", "cob_vac_bcg"]
for estado, df_estado in df_estado.items():
    X = df_estado[feature_cols]
    y = df_estado["ano"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia para o estado {estado}: {acc}")
    dot_data = export_graphviz(
        dtc, out_file=None, feature_names=feature_cols, class_names=anos, rounded=True
    )
    graph = Source(dot_data)
    graph.render(
        f"/home/luiz/Documentos/Iniciacao_cientifica_2022/exported/DT/{estado}_decision_tree"
    )


def old_stuff():
    print("")

    # # Mescla a tabela de analfabetismo com a tabela de CV, renomeia a coluna que contém o ano para Analfabetismo
    # for ano in anos:
    #     dfs = [
    #         df_ufs[ano],
    #         df_analfabetismo[ano].rename(columns={ano: "Analfabetismo"}),
    #         df_renda[ano].rename(columns={ano: "Renda_per_capita"}),
    #     ]

    #     dfs_comparativos[ano] = ft.reduce(
    #         lambda left, right: pd.merge(left, right, on="UF"), dfs
    #     )
    # # Exporta individualmente cada estado brasileiro, separando por ano.
    # # genFile(ano, dfs_comparativos)

    # # Agrupa todos os resultados por Estado.
    # for subdir, _, files in os.walk(diretorio):
    #     for file in files:
    #         if file.endswith(".csv"):
    #             caminho_arquivo = os.path.join(subdir, file)
    #             # obter o nome do estado a partir do nome do arquivo
    #             estado = file.split("-")[0]
    #             # ler a tabela
    #             tabela = pd.read_csv(caminho_arquivo)
    #             # adicionar a coluna "UF" com o nome do estado
    #             tabela["UF"] = estado
    #             # adicionar a tabela no dicionário de UF
    #             if estado in UF:
    #                 UF[estado] = pd.concat([UF[estado], tabela])
    #             else:
    #                 UF[estado] = tabela
    # # Remove a coluna UF
    # for estado, tabela in UF.items():
    #     tabela = tabela.fillna(0)
    #     tabela = tabela.drop(columns=["UF"])
    #     UF[estado] = tabela

    # x = np.array(UF["Sao Paulo"]["Analfabetismo"]).reshape((-1, 1))
    # y = np.array(UF["Sao Paulo"]["BCG"])

    # # https://realpython.com/linear-regression-in-python/

    # model = LinearRegression().fit(x, y)
    # r_sq = model.score(x, y)
    # print(f"R²: {r_sq}")

    # y_pred = model.predict(x)
    # print(x, y)
    # print(UF["Acre"])
    # print(f"predicted response:\n{y_pred}")
