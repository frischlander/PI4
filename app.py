"""
Aplicação Flask principal para o Projeto Integrador IV - Dengue Sertãozinho

- Usa `df_dengue_tratado.csv` para todas as estatísticas e análises gerais.
- Usa `modelo_reglog_pi4.pkl` e o dataset balanceado `df_final_predict.csv` APENAS para o modelo preditivo.
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# --- Carregamento de Dados ---

# 1. Dataset de Sertãozinho (para estatísticas e visualizações)
print("Carregando dataset de Sertãozinho para estatísticas...")
df_stats = pd.read_csv("df_dengue_tratado.csv")
df_stats["DT_NOTIFIC"] = pd.to_datetime(df_stats["DT_NOTIFIC"])
df_stats["DT_SIN_PRI"] = pd.to_datetime(df_stats["DT_SIN_PRI"])
print(f"   ✓ Dataset de estatísticas carregado: {len(df_stats):,} registros")

# 2. Modelo Preditivo
print("Carregando modelo preditivo...")
try:
    model = joblib.load("modelo_reglog_pi4.pkl")
    model_features = model.feature_names_in_
    print(f"   ✓ Modelo carregado: {type(model).__name__}")
    print(f"   ✓ Features do modelo: {len(model_features)}")
except Exception as e:
    print(f"ERRO ao carregar o modelo: {e}")
    model = None
    model_features = []

# --- Rotas da Aplicação ---

@app.route("/")
def index():
    """Renderiza a página principal."""
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    """Renderiza o dashboard interativo."""
    return render_template("dashboard.html")

# --- API para Estatísticas (usa df_stats) ---

@app.route("/api/data/summary")
def data_summary():
    """Retorna um resumo dos dados REAIS de Sertãozinho."""
    total_casos = len(df_stats)
    casos_hospitalizados = len(df_stats[df_stats["HOSPITALIZ"] == "SIM"])
    taxa_hospitalizacao = (casos_hospitalizados / total_casos) * 100 if total_casos > 0 else 0
    
    summary = {
        "total_casos": total_casos,
        "casos_hospitalizados": casos_hospitalizados,
        "taxa_hospitalizacao": round(taxa_hospitalizacao, 2),
        "idade_media": round(df_stats["IDADE"].mean(), 1),
        "anos_cobertura": f"{df_stats['NU_ANO'].min()} - {df_stats['NU_ANO'].max()}"
    }
    return jsonify(summary)

@app.route("/api/data/casos_por_ano")
def casos_por_ano():
    """Retorna dados de casos por ano do dataset de Sertãozinho."""
    casos_ano = df_stats["NU_ANO"].value_counts().sort_index()
    data = {
        "anos": casos_ano.index.tolist(),
        "casos": casos_ano.values.tolist()
    }
    return jsonify(data)

@app.route("/api/data/casos_por_mes")
def casos_por_mes():
    """Retorna dados de casos por mês do dataset de Sertãozinho."""
    df_stats["MES_NOTIFIC"] = df_stats["DT_NOTIFIC"].dt.month
    casos_mes = df_stats["MES_NOTIFIC"].value_counts().sort_index()
    meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    data = {
        "meses": [meses[i-1] for i in casos_mes.index],
        "casos": casos_mes.values.tolist()
    }
    return jsonify(data)

# --- API para Predição (usa o modelo treinado) ---

@app.route("/api/predict", methods=["POST"])
def predict():
    """Endpoint para predição do modelo de Regressão Logística."""
    if model is None:
        return jsonify({"error": "Modelo não carregado"}), 500

    try:
        data = request.json
        
        # Criar um DataFrame com os dados de entrada
        input_df = pd.DataFrame([data])
        
        # 1. Preencher valores ausentes para as colunas que o modelo espera
        default_values = {
            "CS_RACA": "IGNORADO",
            "CS_GESTANT": "NÃO_APLIC",
            "PETEQUIA_N": "NÃO",
            "DIABETES": "NÃO",
            "HEMATOLOG": "NÃO",
            "HEPATOPAT": "NÃO",
            "RENAL": "NÃO",
            "FENOMENO": "Neutro",
            "INTENS_FENOM": "NEUTRA"
        }
        
        for col, default_val in default_values.items():
            if col not in input_df.columns:
                input_df[col] = default_val
        
        # 2. Aplicar one-hot encoding
        categorical_cols = [col for col in ["CS_SEXO", "CS_RACA", "CS_GESTANT", "FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "PETEQUIA_N", "DIABETES", "HEMATOLOG", "HEPATOPAT", "RENAL", "FENOMENO", "INTENS_FENOM"] if col in input_df.columns]
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)
        
        # 3. Alinhar as colunas com as features exatas do modelo
        input_aligned = input_encoded.reindex(columns=model_features, fill_value=0)
        
        # 4. Fazer a predição
        prediction_proba = model.predict_proba(input_aligned)[:, 1]
        
        return jsonify({"probabilidade_hospitalizacao": prediction_proba[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data/distribuicao_sexo")
def distribuicao_sexo():
    """Retorna dados de distribuição por sexo."""
    sexo_counts = df_stats['CS_SEXO'].value_counts()
    data = {
        "labels": sexo_counts.index.tolist(),
        "values": sexo_counts.values.tolist()
    }
    return jsonify(data)

@app.route("/api/data/fenomeno_climatico")
def fenomeno_climatico():
    """Retorna dados de distribuição por fenômeno climático."""
    fenomeno_counts = df_stats['FENOMENO'].value_counts()
    data = {
        "labels": fenomeno_counts.index.tolist(),
        "values": fenomeno_counts.values.tolist()
    }
    return jsonify(data)

@app.route("/api/data/hospitalizacao_por_idade")
def hospitalizacao_por_idade():
    """Retorna dados de hospitalização por faixa etária."""
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 120]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '90+']
    df_stats['faixa_etaria'] = pd.cut(df_stats['IDADE'], bins=bins, labels=labels, right=False)
    
    hosp_por_idade = df_stats[df_stats['HOSPITALIZ'] == 'SIM']['faixa_etaria'].value_counts().sort_index()
    total_por_idade = df_stats['faixa_etaria'].value_counts().sort_index()
    
    data = {
        "faixas": total_por_idade.index.tolist(),
        "hospitalizados": hosp_por_idade.reindex(total_por_idade.index, fill_value=0).values.tolist(),
        "total": total_por_idade.values.tolist()
    }
    return jsonify(data)

@app.route('/api/data/filtered', methods=['POST'])
def get_filtered_data():
    """Retorna dados filtrados para o dashboard."""
    filters = request.json
    
    filtered_df = df_stats.copy()
    
    if filters.get('anoInicial'):
        filtered_df = filtered_df[filtered_df['NU_ANO'] >= int(filters['anoInicial'])]
    if filters.get('anoFinal'):
        filtered_df = filtered_df[filtered_df['NU_ANO'] <= int(filters['anoFinal'])]
    if filters.get('fenomeno'):
        filtered_df = filtered_df[filtered_df['FENOMENO'] == filters['fenomeno']]
    if filters.get('sexo'):
        filtered_df = filtered_df[filtered_df['CS_SEXO'] == filters['sexo']]

    # Recalcular todos os dados para os gráficos com base no df filtrado
    # Summary
    total_casos = len(filtered_df)
    casos_hospitalizados = len(filtered_df[filtered_df["HOSPITALIZ"] == "SIM"])
    taxa_hospitalizacao = (casos_hospitalizados / total_casos) * 100 if total_casos > 0 else 0
    idade_media = round(filtered_df["IDADE"].mean(), 1) if total_casos > 0 else 0

    summary = {
        "total_casos": total_casos,
        "casos_hospitalizados": casos_hospitalizados,
        "taxa_hospitalizacao": round(taxa_hospitalizacao, 2),
        "idade_media": idade_media
    }

    # Casos por Ano
    casos_ano = filtered_df["NU_ANO"].value_counts().sort_index()
    casos_por_ano_data = {
        "anos": casos_ano.index.tolist(),
        "casos": casos_ano.values.tolist()
    }

    # Distribuição por Sexo
    sexo_counts = filtered_df['CS_SEXO'].value_counts()
    distribuicao_sexo_data = {
        "labels": sexo_counts.index.tolist(),
        "values": sexo_counts.values.tolist()
    }

    # Casos por Mês
    filtered_df["MES_NOTIFIC"] = filtered_df["DT_NOTIFIC"].dt.month
    casos_mes = filtered_df["MES_NOTIFIC"].value_counts().sort_index()
    meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    casos_por_mes_data = {
        "meses": [meses[i-1] for i in casos_mes.index],
        "casos": casos_mes.values.tolist()
    }

    # Fenômeno Climático
    fenomeno_counts = filtered_df['FENOMENO'].value_counts()
    fenomeno_climatico_data = {
        "labels": fenomeno_counts.index.tolist(),
        "values": fenomeno_counts.values.tolist()
    }

    # Hospitalização por Idade
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 120]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '90+']
    filtered_df['faixa_etaria'] = pd.cut(filtered_df['IDADE'], bins=bins, labels=labels, right=False)
    hosp_por_idade = filtered_df[filtered_df['HOSPITALIZ'] == 'SIM']['faixa_etaria'].value_counts().sort_index()
    total_por_idade = filtered_df['faixa_etaria'].value_counts().sort_index()
    hospitalizacao_idade_data = {
        "faixas": total_por_idade.index.tolist(),
        "hospitalizados": hosp_por_idade.reindex(total_por_idade.index, fill_value=0).values.tolist(),
        "total": total_por_idade.values.tolist()
    }

    # Distribuição por Raça
    raca_counts = filtered_df['CS_RACA'].value_counts()
    distribuicao_raca_data = {
        "labels": raca_counts.index.tolist(),
        "values": raca_counts.values.tolist()
    }

    return jsonify({
        "summary": summary,
        "casosPorAno": casos_por_ano_data,
        "distribuicaoSexo": distribuicao_sexo_data,
        "casosPorMes": casos_por_mes_data,
        "fenomenoClimatico": fenomeno_climatico_data,
        "hospitalizacaoIdade": hospitalizacao_idade_data,
        "racaDistribution": distribuicao_raca_data
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
