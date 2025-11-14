import joblib
import pickle

model = joblib.load('modelo_reglog_pi4.pkl')

print("Informações do modelo:")
print(f"Tipo: {type(model).__name__}")
print(f"Número de features: {model.coef_.shape[1]}")

# Tentar obter feature names
if hasattr(model, 'feature_names_in_'):
    print(f"\nFeature names do modelo:")
    for i, feat in enumerate(model.feature_names_in_, 1):
        print(f"{i:3d}. {feat}")
    
    # Salvar em arquivo
    with open('model_features.txt', 'w') as f:
        for feat in model.feature_names_in_:
            f.write(f"{feat}\n")
    print(f"\n✓ Features salvas em model_features.txt")
else:
    print("\nModelo não tem feature_names_in_")
