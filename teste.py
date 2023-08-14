import pandas as pd

# Crie a primeira tabela
tabela1 = pd.DataFrame(
    {"UF": ["SP", "RJ", "MG"], "População": [45000000, 17000000, 21000000]}
)

# Crie a segunda tabela
tabela2 = pd.DataFrame(
    {"UF": ["SP", "MG", "PR"], "População": [46000000, 22000000, 11000000]}
)

# Concatene as duas tabelas
tabela_concatenada = pd.concat([tabela1, tabela2], ignore_index=True)

# Imprima a tabela concatenada
print(tabela_concatenada)
