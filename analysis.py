import pandas.io.sql as sqlio
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

conn = psycopg2.connect(
    "dbname = 'postgres' user = 'postgres' host = '192.168.15.45' port = '7777' password = 'ic_2023'"
)


def get2019():
    query = 'select * from "total_2019";'
    return sqlio.read_sql_query(query, conn)


def get2020():
    query = 'select * from "total_2020";'
    return sqlio.read_sql_query(query, conn)


def get2021():
    query = 'select * from "total_2021";'
    return sqlio.read_sql_query(query, conn)


def getSudesteEvo():
    query = 'select * from "sudeste_evo";'
    return sqlio.read_sql_query(query, conn)


def getCentroOesteEvo():
    query = 'select * from "centro_oeste_evo";'
    return sqlio.read_sql_query(query, conn)


def getNorteEvo():
    query = 'select * from "norte_evo";'
    return sqlio.read_sql_query(query, conn)


def getNordesteEvo():
    query = 'select * from "nordeste_evo";'
    return sqlio.read_sql_query(query, conn)


def getSulEvo():
    query = 'select * from "sul_evo";'
    return sqlio.read_sql_query(query, conn)


def getUF2019():
    query = 'select * from "uf_2019";'
    return sqlio.read_sql_query(query, conn)


def getUF2020():
    query = 'select * from "uf_2020";'
    return sqlio.read_sql_query(query, conn)


def getUF2021():
    query = 'select * from "uf_2021";'
    return sqlio.read_sql_query(query, conn)


def getAnalfabetismo(year):
    query = 'select "UF","' + year + '" from "analfabetismo2019";'
    return sqlio.read_sql_query(query, conn)


df_2019 = getUF2019()
df_2020 = getUF2020()
df_2021 = getUF2021()
df_analfabetismo_2019 = getAnalfabetismo("2019")
df_analfabetismo_2020 = getAnalfabetismo("2020")
df_analfabetismo_2021 = getAnalfabetismo("2021")

df_2019_comparative = pd.merge(df_2019, df_analfabetismo_2019, on="UF")
df_2020_comparative = pd.merge(df_2020, df_analfabetismo_2020, on="UF")
df_2021_comparative = pd.merge(df_2021, df_analfabetismo_2021, on="UF")

# UF_grouped = df_2019_comparative.groupby("UF")
# for uf, grupo in UF_grouped:
#     tabela_uf = grupo.copy()
#     tabela_uf["UF"] = tabela_uf["UF"].apply(lambda x: f"{x} 2019")
#     tabela_uf.to_csv(f"./dados/Filtered/2019/{uf}-2019.csv", index=False)


Acre2019 = pd.read_csv("./dados/Filtered/2019/Acre-2019.csv")
Acre2020 = pd.read_csv("./dados/Filtered/2020/Acre-2020.csv")
Acre2021 = pd.read_csv("./dados/Filtered/2021/Acre-2021.csv")

Acre = Acre2019.append((Acre2020, Acre2021), ignore_index=True)
profile = ProfileReport(Acre, title="report_acre")
profile.to_file("report.html")
