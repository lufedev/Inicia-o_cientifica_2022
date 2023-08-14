import sqlite3
import pandas as pd
import numpy
import sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from ydata_profiling import ProfileReport
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

numpy.set_printoptions(threshold=sys.maxsize)


def load_data(
    database_file, region_columns, vaccine_columns, other_columns, target_column
):
    """Carrega os dados do banco de dados SQLite e seleciona as colunas relevantes."""
    conn = sqlite3.connect(database_file)
    str_format = f'"{region_columns[1]}"'
    query = f"SELECT {', '.join(vaccine_columns + other_columns + [target_column])} FROM raw_data WHERE {region_columns[0]} = '{region_columns[1]}'"
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def create_decision_tree_model(X_train, X_test, y_train, y_test):
    """Cria e treina o modelo DecisionTreeClassifier."""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Fazendo previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliando a precisão do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy:.2f}")

    return model


if __name__ == "__main__":
    # Substitua 'nome_do_banco_de_dados.db' pelo nome do seu arquivo SQLite
    database_file = "./dados/raw/sqlite_data"

    # Substitua 'nome_da_tabela' pelo nome da tabela que contém os dados
    table_name = "raw_data"

    # Colunas que informam a região
    region_columns = ["regiao_geo", "Sul"]

    # Colunas que informam as vacinas
    vaccine_columns = [
        "exp_vida",
        "idhm_educ",
        "idhm_renda",
        "pct_san_adeq",
        "pop",
        "n_obitos",
        "renda_dom_pc",
    ]

    # Colunas que informam outros índices (exceto o target)
    other_columns = []

    # Coluna que informa o target (cobertura vacinal da BCG)
    target_column = "cob_vac_polio"

    # Carrega os dados e seleciona as colunas relevantes
    data = load_data(
        database_file, region_columns, vaccine_columns, other_columns, target_column
    )

    # Separa as features (X) e o target (y)
    X = data[vaccine_columns + other_columns]
    Y = data[target_column]

    # Normaliza as colunas (exceto o target) usando MinMaxScaler
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    data = pd.DataFrame(X_normalized, columns=X.columns)
    RegionProfile = ProfileReport(
        data, title="Indices da região selecionada", explorative=True
    )

    print(RegionProfile)
    RegionProfile.to_file("RegionProfile.html")
    #   Separa os dados em conjuntos de treino e teste

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=42, max_depth=10, min_samples_leaf=12
    )
    clf = clf_entropy.fit(X_train, y_train)

    y_pred = clf_entropy.predict(X_test)
    # print(y_pred)
    print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

    dot_data = StringIO()
    export_graphviz(
        clf,
        out_file=dot_data,
        filled=True,
        rounded=True,
        special_characters=True,
        feature_names=vaccine_columns,
        class_names=["0", "1", "2", "3"],
    )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("cobertura.png")
    Image(graph.create_png())
