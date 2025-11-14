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

# --- Funções de Geração de Imagens ---

def generate_eda_charts(df):
    """Gera gráficos de Análise Exploratória de Dados (EDA) do df_dengue_tratado."""
    
    # 1. Casos por Ano
    df['NU_ANO'] = df['DT_NOTIFIC'].apply(lambda x: pd.to_datetime(x).year)
    casos_ano = df['NU_ANO'].value_counts().sort_index()
    plt.figure()
    casos_ano.plot(kind='bar', color='#667eea')
    plt.title('Casos de Dengue por Ano')
    plt.xlabel('Ano')
    plt.ylabel('Número de Casos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'casos_por_ano.png'))
    plt.close()

    # 2. Distribuição por Sexo
    sexo_counts = df['CS_SEXO'].value_counts()
    plt.figure()
    plt.pie(sexo_counts, labels=sexo_counts.index, autopct='%1.1f%%', colors=['#f093fb', '#667eea'])
    plt.title('Distribuição de Casos por Sexo')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sexo_distribuicao.png'))
    plt.close()

    # 3. Distribuição por Idade
    plt.figure()
    sns.histplot(df['IDADE'], bins=20, kde=True, color='#667eea')
    plt.title('Distribuição de Casos por Idade')
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'idade_distribuicao.png'))
    plt.close()

    # 4. Distribuição por Raça
    raca_counts = df['CS_RACA'].value_counts()
    plt.figure()
    raca_counts.plot(kind='barh', color='#ff9f40')
    plt.title('Distribuição de Casos por Raça')
    plt.xlabel('Número de Casos')
    plt.ylabel('Raça')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'raca_distribuicao.png'))
    plt.close()

    # 5. Fenômeno Climático
    fenomeno_counts = df['FENOMENO'].value_counts()
    plt.figure()
    plt.pie(fenomeno_counts, labels=fenomeno_counts.index, autopct='%1.1f%%', colors=['#ffecd2', '#a8edea', '#f093fb'])
    plt.title('Distribuição de Casos por Fenômeno Climático')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fenomeno_climatico_distribuicao.png'))
    plt.close()

    # 6. Intensidade do Fenômeno
    intensidade_counts = df['INTENS_FENOM'].value_counts()
    plt.figure()
    intensidade_counts.plot(kind='bar', color='#a8edea')
    plt.title('Distribuição de Casos por Intensidade do Fenômeno')
    plt.xlabel('Intensidade')
    plt.ylabel('Número de Casos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intensidade_fenomeno_distribuicao.png'))
    plt.close()

def generate_model_charts(model, df_predict):
    """Gera gráficos de Análise do Modelo (Matriz de Confusão, Importância de Features, Curva ROC)."""
    
    # 1. Aplicar One-Hot Encoding no df_predict para alinhamento
    categorical_cols = ["CS_SEXO", "CS_RACA", "CS_GESTANT", "FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "PETEQUIA_N", "DIABETES", "HEMATOLOG", "HEPATOPAT", "RENAL", "FENOMENO", "INTENS_FENOM"]
    df_encoded = pd.get_dummies(df_predict, columns=categorical_cols, drop_first=False)
    
    # Identificar features e alvo
    model_features = model.feature_names_in_
    target_col = 'HOSPITALIZ_SIM'
    
    # Alinhar as colunas com as features exatas do modelo
    X = df_encoded.reindex(columns=model_features, fill_value=0)
    
    # Criar a coluna alvo para fins de visualização (ela deve existir no df_predict original)
    # Como o df_final_predict.csv é o dataset balanceado, ele deve ter a coluna alvo.
    # Vamos assumir que a coluna alvo é 'HOSPITALIZ_SIM' (resultado do one-hot encoding da coluna 'HOSPITALIZ')
    # Como o df_predict não tem a coluna alvo, vamos criá-la a partir da coluna 'HOSPITALIZ'
    # do df_predict, que deve ter 'SIM' ou 'NÃO'.
    
    # O df_final_predict.csv não tem a coluna alvo, o que é um problema.
    # Para fins de visualização, vamos usar a coluna 'HOSPITALIZ' do df_predict para criar o alvo
    # e assumir que o modelo foi treinado para prever 'SIM' (1) ou 'NÃO' (0).
    
    # O df_final_predict.csv não tem a coluna alvo, o que é um problema.
    # Para fins de visualização, vamos usar a coluna 'HOSPITALIZ' do df_predict para criar o alvo
    # e assumir que o modelo foi treinado para prever 'SIM' (1) ou 'NÃO' (0).
    
    # Vamos criar a coluna alvo 'HOSPITALIZ_SIM' a partir da coluna 'HOSPITALIZ'
    if 'HOSPITALIZ' in df_predict.columns:
        y = (df_predict['HOSPITALIZ'] == 'SIM').astype(int)
    else:
        # Se a coluna 'HOSPITALIZ' não existir, não podemos gerar os gráficos de métricas
        print("AVISO: Coluna 'HOSPITALIZ' não encontrada no df_final_predict. Não é possível gerar gráficos de métricas.")
        return

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
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. Importância das Features (Coeficientes da Regressão Logística)
    if hasattr(model, 'coef_'):
        coefs = pd.Series(model.coef_[0], index=model_features)
        top_n = 15
        # Garantir que temos pelo menos 15 coeficientes
        if len(coefs) < top_n:
            top_n = len(coefs)
        
        top_coefs = pd.concat([coefs.nlargest(top_n//2), coefs.nsmallest(top_n//2)])
        
        plt.figure()
        top_coefs.sort_values().plot(kind='barh', color=top_coefs.apply(lambda x: 'red' if x < 0 else 'green'))
        plt.title(f'Top {top_n} Coeficientes do Modelo (Importância das Features)')
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
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

# --- Execução Principal ---

def main():
    # Carregar DataFrames
    try:
        df_stats = pd.read_csv("df_dengue_tratado.csv")
        df_predict = pd.read_csv("df_final_predict.csv")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo de dados não encontrado: {e}")
        return

    # Carregar Modelo
    try:
        model = joblib.load("modelo_reglog_pi4.pkl")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo do modelo não encontrado: {e}")
        return

    # 1. Gerar Gráficos EDA
    print("Gerando gráficos de Análise Exploratória de Dados...")
    generate_eda_charts(df_stats)
    print("   ✓ Gráficos EDA gerados em static/images/")

    # 2. Gerar Gráficos do Modelo
    print("Gerando gráficos de Análise do Modelo...")
    generate_model_charts(model, df_predict)
    print("   ✓ Gráficos do Modelo gerados em static/images/")

if __name__ == "__main__":
    # Mudar para o diretório do projeto
    os.chdir('/home/ubuntu/PI4')
    main()
