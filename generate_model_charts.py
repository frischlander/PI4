"""
Script para gerar gráficos de análise do modelo preditivo
Usa o dataset balanceado (df_final_predict.csv) apenas para treinar e avaliar o modelo
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Configurações de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*80)
print("GERANDO GRÁFICOS DE ANÁLISE DO MODELO PREDITIVO")
print("="*80)

# Carregar modelo
print("\n1. Carregando modelo...")
model = joblib.load('modelo_reglog_pi4.pkl')
print(f"   ✓ Modelo carregado: {type(model).__name__}")

# Carregar dataset balanceado (APENAS para o modelo)
print("\n2. Carregando dataset balanceado...")
df = pd.read_csv('df_final_predict.csv')
print(f"   ✓ Dataset carregado: {len(df):,} registros")

# Preparar os dados EXATAMENTE como foram usados no treinamento
print("\n3. Preparando dados para avaliação...")
categorical_cols = ['CS_SEXO', 'CS_RACA', 'FEBRE', 'MIALGIA', 'CEFALEIA', 
                   'EXANTEMA', 'VOMITO', 'PETEQUIA_N', 'DIABETES', 'HEMATOLOG', 
                   'HEPATOPAT', 'RENAL', 'FENOMENO', 'INTENS_FENOM']

# Aplicar one-hot encoding SEM drop_first para manter todas as categorias
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Usar as features exatas do modelo
model_features = model.feature_names_in_
X = df_encoded[model_features]
y = df['HOSPITALIZ'].apply(lambda x: 1 if x == 'SIM' else 0)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   ✓ Treino: {len(X_train):,} | Teste: {len(X_test):,}")

# Fazer predições
print("\n4. Fazendo predições...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calcular métricas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n   Métricas do Modelo:")
print(f"   - Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   - Precisão: {precision:.4f} ({precision*100:.2f}%)")
print(f"   - Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"   - F1-Score: {f1:.4f} ({f1*100:.2f}%)")
print(f"   - ROC AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")

# Salvar métricas em arquivo
with open('model_metrics.txt', 'w') as f:
    f.write("MÉTRICAS DO MODELO PREDITIVO\n")
    f.write("="*80 + "\n\n")
    f.write(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precisão: {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall: {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)\n")
    f.write(f"ROC AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)\n")

# 1. Matriz de Confusão
print("\n5. Gerando Matriz de Confusão...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Hospitalizado', 'Hospitalizado'], 
            yticklabels=['Não Hospitalizado', 'Hospitalizado'],
            cbar_kws={'label': 'Contagem'})
plt.ylabel('Verdadeiro', fontsize=12, fontweight='bold')
plt.xlabel('Predito', fontsize=12, fontweight='bold')
plt.title('Matriz de Confusão - Modelo de Regressão Logística', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('static/images/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: static/images/confusion_matrix.png")
plt.close()

# 2. Importância das Features (Coeficientes)
print("6. Gerando gráfico de Importância das Features...")
coefficients = pd.DataFrame({
    'Feature': model_features,
    'Coefficient': model.coef_[0]
})
coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values(by='Abs_Coefficient', ascending=False).head(15)

plt.figure(figsize=(12, 8))
colors = ['red' if x < 0 else 'green' for x in coefficients['Coefficient']]
plt.barh(range(len(coefficients)), coefficients['Coefficient'], color=colors, edgecolor='black')
plt.yticks(range(len(coefficients)), coefficients['Feature'])
plt.xlabel('Coeficiente', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 15 Features Mais Importantes (Coeficientes da Regressão Logística)', 
          fontsize=14, fontweight='bold', pad=20)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/feature_importance.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: static/images/feature_importance.png")
plt.close()

# 3. Curva ROC
print("7. Gerando Curva ROC...")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Linha de Referência')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos', fontsize=12, fontweight='bold')
plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12, fontweight='bold')
plt.title('Curva ROC - Modelo de Regressão Logística', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/roc_curve.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: static/images/roc_curve.png")
plt.close()

# 4. Salvar Relatório de Classificação
print("8. Salvando relatório de classificação...")
report = classification_report(y_test, y_pred, target_names=['Não Hospitalizado', 'Hospitalizado'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report.csv')
print("   ✓ Salvo: classification_report.csv")

print("\n" + "="*80)
print("TODOS OS GRÁFICOS DO MODELO FORAM GERADOS COM SUCESSO!")
print("="*80)
print(f"\nMétricas Finais:")
print(f"  - ROC AUC: {roc_auc*100:.2f}%")
print(f"  - Acurácia: {accuracy*100:.2f}%")
print(f"  - Precisão: {precision*100:.2f}%")
print(f"  - Recall: {recall*100:.2f}%")
print("\nArquivos gerados:")
print("  - static/images/confusion_matrix.png")
print("  - static/images/feature_importance.png")
print("  - static/images/roc_curve.png")
print("  - model_metrics.txt")
print("  - classification_report.csv")
