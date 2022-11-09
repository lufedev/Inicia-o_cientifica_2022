import sys
import csv
import pandas.io.sql as sqlio
from pandas_profiling import ProfileReport
import psycopg2

conn = psycopg2.connect("dbname = 'postgres' user = 'postgres' host = '192.168.15.45' port = '7777' password = 'ic_2022'")
#driving_distance module
#note the lack of trailing semi-colon in the query string, as per the Postgres documentation
query = "select * from \"2021\";"
  
df = sqlio.read_sql_query(query, conn)

profile = ProfileReport(df, title="Profiling Report", explorative=True)
profile.to_file("2021.html")
