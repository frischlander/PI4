import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuração de estilo e diretórios
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
output_dir = 'static/images'
os.makedirs(output_dir, exist_ok=True)

def generate_feature_importance(model):
    """Gera o gráfico de Importância das Features (Coeficientes da Regressão Logística)."""
    
    if hasattr(model, 'coef_'):
        model_features = model.feature_names_in_
        coefs = pd.Series(model.coef_[0], index=model_features)
        
        # Filtrar apenas as features de input
        input_features_base = ['IDADE', 'CS_SEXO', 'FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']
        
        # Filtrar os coeficientes que correspondem às features de input (incluindo one-hot encoding)
        filtered_coefs = coefs[coefs.index.str.contains('|'.join(input_features_base))]
        
        # Se o modelo foi treinado com todas as 47 features, vamos mostrar as 15 mais importantes
        # que contêm as features de input.
        
        # Para o gráfico de importância, vamos mostrar os 15 coeficientes mais relevantes
        top_n = 15
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
    else:
        print("Modelo não possui atributo 'coef_' para gerar o gráfico de importância.")

# --- Execução Principal ---

def main():
    # Carregar Modelo
    try:
        model = joblib.load("modelo_reglog_pi4.pkl")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo do modelo não encontrado: {e}")
        return

    # Gerar Gráfico de Importância
    print("Gerando novo gráfico de Importância das Features...")
    generate_feature_importance(model)
    print("   ✓ Gráfico de Importância das Features gerado em static/images/feature_importance.png")

if __name__ == "__main__":
    # Mudar para o diretório do projeto
    os.chdir('/home/ubuntu/PI4')
    main()
