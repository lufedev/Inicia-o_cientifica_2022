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


getSudesteEvo().plot.bar()
getNordesteEvo().plot.bar()
plt.show()
