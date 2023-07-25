import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


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
    region_columns = ["estado", "Bahia"]

    # Colunas que informam as vacinas
    vaccine_columns = ["exp_vida", "idhm"]

    # Colunas que informam outros índices (exceto o target)
    other_columns = []

    # Coluna que informa o target (cobertura vacinal da BCG)
    target_column = "cob_vac_bcg"

    # Carrega os dados e seleciona as colunas relevantes
    data = load_data(
        database_file, region_columns, vaccine_columns, other_columns, target_column
    )

    # Separa as features (X) e o target (y)
    X = data[vaccine_columns + other_columns]
    y = data[target_column]

    # Normaliza as colunas (exceto o target) usando MinMaxScaler
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Divide os dados normalizados em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42
    )

    # Cria e treina o modelo DecisionTreeClassifier
    model = create_decision_tree_model(X_train, X_test, y_train, y_test)

    # Agora você pode usar o modelo treinado para fazer previsões em novos dados normalizados.
    # Lembre-se de normalizar os novos dados usando o mesmo scaler que você usou nos dados de treinamento.
    # Por exemplo:
    new_data = pd.DataFrame([[72, 0.800]], columns=["exp_vida", "idhm"])
    new_data_normalized = scaler.transform(new_data)
    prediction = model.predict(new_data_normalized)
    print("Previsão:", prediction)
