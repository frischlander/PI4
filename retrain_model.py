import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import numpy as np

# Definir as 7 features de input
INPUT_FEATURES = ['IDADE', 'CS_SEXO', 'FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']
TARGET_COLUMN = 'HOSPITALIZ'

def retrain_model():
    print("Iniciando retreinamento do modelo...")
    
    # 1. Carregar o dataset balanceado
    try:
        df = pd.read_csv("df_final_predict.csv")
    except FileNotFoundError:
        print("ERRO: df_final_predict.csv não encontrado.")
        return

    # 2. Pré-processamento e Feature Engineering
    
    # Selecionar apenas as features de input e a coluna alvo
    df_model = df[INPUT_FEATURES + [TARGET_COLUMN]].copy()
    
    # Aplicar One-Hot Encoding nas features categóricas
    categorical_cols = ['CS_SEXO', 'FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=False)
    
    # Criar a coluna alvo binária (HOSPITALIZ_SIM)
    df_encoded['HOSPITALIZ_SIM'] = (df_encoded[TARGET_COLUMN] == 'SIM').astype(int)
    
    # Remover a coluna alvo original e outras colunas desnecessárias
    X = df_encoded.drop(columns=[TARGET_COLUMN, 'HOSPITALIZ_SIM'])
    y = df_encoded['HOSPITALIZ_SIM']
    
    # 3. Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Treinar o Modelo
    print(f"Treinando modelo com {len(X.columns)} features...")
    model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Avaliação
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=['NÃO', 'SIM'], output_dict=True)
    
    print("\n--- Resultados do Modelo Retreinado ---")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Acurácia: {report['accuracy']:.4f}")
    print(f"Recall (SIM): {report['SIM']['recall']:.4f}")
    print(f"Precisão (SIM): {report['SIM']['precision']:.4f}")
    
    # 6. Salvar o Novo Modelo
    new_model_path = "modelo_reglog_pi4_retrained.pkl"
    joblib.dump(model, new_model_path)
    print(f"\n✓ Novo modelo salvo em: {new_model_path}")
    
    # 7. Salvar as métricas para uso na interface
    metrics = {
        "roc_auc": round(roc_auc * 100, 2),
        "accuracy": round(report['accuracy'] * 100, 2),
        "recall_sim": round(report['SIM']['recall'] * 100, 2),
        "precision_sim": round(report['SIM']['precision'] * 100, 2),
        "features": X.columns.tolist()
    }
    
    with open("model_metrics_retrained.json", "w") as f:
        import json
        json.dump(metrics, f)
    
    print("✓ Métricas salvas em model_metrics_retrained.json")

if __name__ == "__main__":
    # Mudar para o diretório do projeto
    import os
    os.chdir('/home/ubuntu/PI4')
    retrain_model()
