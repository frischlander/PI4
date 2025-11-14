import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import os

# Configuração de estilo e diretórios
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
output_dir = 'static/images'
os.makedirs(output_dir, exist_ok=True)

# Definir as 7 features de input
INPUT_FEATURES = ['IDADE', 'CS_SEXO', 'FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']
TARGET_COLUMN = 'HOSPITALIZ'

def generate_model_charts(model, df_predict):
    """Gera gráficos de Análise do Modelo (Matriz de Confusão, Importância de Features, Curva ROC)."""
    
    # 1. Preparação dos dados de teste
    
    # Aplicar One-Hot Encoding nas features categóricas
    categorical_cols = ['CS_SEXO', 'FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']
    df_model = df_predict[INPUT_FEATURES + [TARGET_COLUMN]].copy()
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=False)
    
    # Criar a coluna alvo binária
    df_encoded['HOSPITALIZ_SIM'] = (df_encoded[TARGET_COLUMN] == 'SIM').astype(int)
    
    # Remover a coluna alvo original e outras colunas desnecessárias
    X = df_encoded.drop(columns=[TARGET_COLUMN, 'HOSPITALIZ_SIM'])
    y = df_encoded['HOSPITALIZ_SIM']
    
    # Dividir em treino e teste (simulação)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 1. Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não Hosp.', 'Hosp.'], yticklabels=['Não Hosp.', 'Hosp.'])
    plt.title('Matriz de Confusão (Modelo Retreinado)')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. Importância das Features (Coeficientes da Regressão Logística)
    if hasattr(model, 'coef_'):
        model_features = X.columns # Usar as colunas do X_test
        coefs = pd.Series(model.coef_[0], index=model_features)
        
        plt.figure()
        coefs.sort_values().plot(kind='barh', color=coefs.apply(lambda x: 'red' if x < 0 else 'green'))
        plt.title('Coeficientes do Modelo (Importância das Features)')
        plt.xlabel('Coeficiente')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()

    # 3. Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='#667eea', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo (1 - Especificidade)')
    plt.ylabel('Taxa de Verdadeiro Positivo (Sensibilidade)')
    plt.title('Curva ROC (Modelo Retreinado)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

# --- Execução Principal ---

def main():
    # Carregar DataFrames
    try:
        df_predict = pd.read_csv("df_final_predict.csv")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo de dados não encontrado: {e}")
        return

    # Carregar Modelo Retreinado
    try:
        model = joblib.load("modelo_reglog_pi4_retrained.pkl")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo do modelo retreinado não encontrado: {e}")
        return

    # Gerar Gráficos do Modelo
    print("Gerando gráficos de Análise do Modelo Retreinado...")
    generate_model_charts(model, df_predict)
    print("   ✓ Gráficos do Modelo gerados em static/images/")

if __name__ == "__main__":
    # Mudar para o diretório do projeto
    os.chdir('/home/ubuntu/PI4')
    main()
