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

anos = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
df_ufs = {}
df_analfabetismo = {}
df_renda = {}
dfs_comparativos = {}
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


# def genFile(ano, dfs_comparativos):
#     for ano, df_comparativo in dfs_comparativos.items():
#         UF_grouped = df_comparativo.groupby("UF")
#     for uf, grupo in UF_grouped:
#         tabela_uf = grupo.copy()
#         tabela_uf["UF"] = tabela_uf["UF"].apply(lambda x: f"{x} {ano}")
#         tabela_uf.to_csv(f"./dados/Filtered/{ano}/{uf}-{ano}.csv", index=False)


dfAno = getByYear("2010")
colunas = []
for coluna in dfAno.columns:
    if (
        coluna == "regiao_geo"
        or coluna == "nomemun"
        or coluna == "capital"
        or coluna == "id_munic_7"
        or coluna == "id_estado"
        or coluna == "estado_abrev"
        or coluna == "estado"
        or coluna == "regiao"
        or coluna == "no_regiao"
        or coluna == "macrorregiao"
        or coluna == "no_macro"
    ):
        continue  # pular a coluna3
    colunas.append(coluna)

X = dfAno[colunas]
y = dfAno["id_estado"]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)  # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# dot_data = StringIO()
# export_graphviz(
#     clf,
#     out_file=dot_data,
#     filled=True,
#     rounded=True,
#     special_characters=True,
#     feature_names=colunas,
#     class_names=["0", "1"],
# )
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png("cv_vacinal.png")
# Image(graph.create_png())


def old_stuff():
    print("")
    # # Coleta a cv anual
    # for ano in anos:
    #     df_ufs[ano] = getUF(ano)
    #     df_analfabetismo[ano] = getAnalfabetismo(ano)
    #     df_renda[ano] = getRenda(ano)

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
