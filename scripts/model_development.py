
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib

# Carregar os datasets pré-processados
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze() # .squeeze() para converter DataFrame de 1 coluna em Series
y_test = pd.read_csv("y_test.csv").squeeze()

print("Dados de treino e teste carregados.")
print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de y_train: {y_train.shape}")

# --- Treinamento do Modelo: Regressão Logística ---
print("\nTreinando Modelo de Regressão Logística...")
log_reg_model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced') # class_weight para lidar com desbalanceamento
log_reg_model.fit(X_train, y_train)

# Avaliação do Modelo de Regressão Logística
y_pred_log_reg = log_reg_model.predict(X_test)
y_proba_log_reg = log_reg_model.predict_proba(X_test)[:, 1]

print("\n--- Avaliação do Modelo de Regressão Logística ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print(f"Precisão: {precision_score(y_test, y_pred_log_reg):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_log_reg):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_log_reg):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_log_reg):.4f}")
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred_log_reg))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_log_reg))

# --- Treinamento do Modelo: Random Forest Classifier ---
print("\nTreinando Modelo Random Forest Classifier...")
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Avaliação do Modelo Random Forest Classifier
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Avaliação do Modelo Random Forest Classifier ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precisão: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred_rf))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_rf))

# Salvar o melhor modelo (ex: Random Forest, se tiver melhor performance)
# Para este exemplo, vamos salvar o Random Forest
joblib.dump(rf_model, "random_forest_model.pkl")
print("\nModelo Random Forest salvo como random_forest_model.pkl")

# Salvar o scaler usado para a idade
# (Assumindo que o scaler foi treinado no script de feature engineering e que X_train["IDADE"] foi escalado)
# Para simplificar, para este exemplo, vamos focar apenas no modelo.
# Em um cenário real, o scaler também seria salvo e carregado.

print("Desenvolvimento e avaliação de modelos concluídos.")

