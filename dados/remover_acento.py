import unidecode
import csv
import sys

feed = sys.argv[1]
csv_f = open(feed, mode="r")
csv_str = csv_f.read()
csv_str_removed_accent = unidecode.unidecode(csv_str)
csv_f.close()
csv_f = open(feed, "w")
csv_f.write(csv_str_removed_accent)
