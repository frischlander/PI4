
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Carregar o dataset
df = pd.read_csv("df_dengue_tratado.csv")

# Converter colunas de data para o tipo datetime (se ainda não foram)
df["DT_NOTIFIC"] = pd.to_datetime(df["DT_NOTIFIC"])
df["DT_SIN_PRI"] = pd.to_datetime(df["DT_SIN_PRI"])

# Definir a variável alvo
# Mapear 'SIM' para 1 e 'NAO' para 0
df["HOSPITALIZ_TARGET"] = df["HOSPITALIZ"].apply(lambda x: 1 if x == "SIM" else 0)

# Selecionar features relevantes para o modelo
# Excluir colunas de identificação, datas originais e a coluna 'HOSPITALIZ' original
# Manter as colunas de sintomas, comorbidades e fenômenos climáticos
features = [
    "IDADE", "CS_SEXO", "CS_RACA", "CS_GESTANT",
    "FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "PETEQUIA_N",
    "DIABETES", "HEMATOLOG", "HEPATOPAT", "RENAL",
    "FENOMENO", "INTENS_FENOM"
]

X = df[features]
y = df["HOSPITALIZ_TARGET"]

# Tratar variáveis categóricas
# Identificar colunas categóricas
categorical_cols = X.select_dtypes(include=["object"]).columns

# Aplicar One-Hot Encoding para as variáveis categóricas
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalar variáveis numéricas (apenas IDADE neste caso)
# Apenas se houver variáveis numéricas que não sejam binárias após o one-hot encoding
numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

# Se 'IDADE' estiver entre as colunas numéricas, aplicar StandardScaler
if 'IDADE' in numerical_cols:
    scaler = StandardScaler()
    X_train['IDADE'] = scaler.fit_transform(X_train[['IDADE']])
    X_test['IDADE'] = scaler.transform(X_test[['IDADE']])

# Salvar os datasets pré-processados para uso posterior
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Feature Engineering e pré-processamento concluídos.")
print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Primeiras 5 linhas de X_train (após pré-processamento):\n{X_train.head()}")

