"""
Script para gerar todos os gráficos e visualizações da aplicação
Usa o dataset de Sertãozinho (df_dengue_tratado.csv) para estatísticas reais
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configurações de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Carregar o dataset de Sertãozinho
print("Carregando dataset de Sertãozinho...")
df = pd.read_csv('df_dengue_tratado.csv')
df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'])
df['DT_SIN_PRI'] = pd.to_datetime(df['DT_SIN_PRI'])

print(f"Dataset carregado: {len(df):,} registros")
print(f"Período: {df['NU_ANO'].min()} - {df['NU_ANO'].max()}")

# 1. Casos por Ano
print("\n1. Gerando gráfico de casos por ano...")
plt.figure(figsize=(14, 6))
casos_ano = df['NU_ANO'].value_counts().sort_index()
plt.bar(casos_ano.index, casos_ano.values, color='steelblue', edgecolor='black')
plt.xlabel('Ano', fontsize=12, fontweight='bold')
plt.ylabel('Número de Casos', fontsize=12, fontweight='bold')
plt.title('Distribuição de Casos de Dengue por Ano em Sertãozinho-SP (2000-2025)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/casos_por_ano.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: static/images/casos_por_ano.png")
plt.close()

# 2. Casos por Mês e Ano (Heatmap)
print("2. Gerando heatmap de casos por mês e ano...")
df['MES'] = df['DT_NOTIFIC'].dt.month
df['ANO'] = df['DT_NOTIFIC'].dt.year
casos_mes_ano = df.groupby(['ANO', 'MES']).size().unstack(fill_value=0)

plt.figure(figsize=(16, 10))
sns.heatmap(casos_mes_ano, cmap='YlOrRd', annot=False, fmt='d', cbar_kws={'label': 'Número de Casos'})
plt.xlabel('Mês', fontsize=12, fontweight='bold')
plt.ylabel('Ano', fontsize=12, fontweight='bold')
plt.title('Distribuição Mensal de Casos de Dengue por Ano', fontsize=14, fontweight='bold', pad=20)
meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
plt.xticks(range(12), meses, rotation=0)
plt.tight_layout()
plt.savefig('static/images/casos_por_mes_ano.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: static/images/casos_por_mes_ano.png")
plt.close()

# 3. Distribuição por Sexo
print("3. Gerando gráfico de distribuição por sexo...")
plt.figure(figsize=(10, 6))
sexo_counts = df['CS_SEXO'].value_counts()
colors = ['#3498db', '#e74c3c']
plt.pie(sexo_counts.values, labels=sexo_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=colors, textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Distribuição de Casos por Sexo', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('static/images/sexo_distribuicao.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: static/images/sexo_distribuicao.png")
plt.close()

# 4. Distribuição por Idade
print("4. Gerando gráfico de distribuição por idade...")
plt.figure(figsize=(12, 6))
plt.hist(df['IDADE'], bins=50, color='teal', edgecolor='black', alpha=0.7)
plt.xlabel('Idade', fontsize=12, fontweight='bold')
plt.ylabel('Frequência', fontsize=12, fontweight='bold')
plt.title('Distribuição de Casos por Idade', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/idade_distribuicao.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: static/images/idade_distribuicao.png")
plt.close()

# 5. Distribuição por Raça
print("5. Gerando gráfico de distribuição por raça...")
plt.figure(figsize=(12, 6))
raca_counts = df['CS_RACA'].value_counts()
plt.barh(raca_counts.index, raca_counts.values, color='coral', edgecolor='black')
plt.xlabel('Número de Casos', fontsize=12, fontweight='bold')
plt.ylabel('Raça', fontsize=12, fontweight='bold')
plt.title('Distribuição de Casos por Raça', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/raca_distribuicao.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: static/images/raca_distribuicao.png")
plt.close()

# 6. Distribuição por Fenômeno Climático
print("6. Gerando gráfico de distribuição por fenômeno climático...")
plt.figure(figsize=(10, 6))
fenomeno_counts = df['FENOMENO'].value_counts()
colors_fenomeno = ['#3498db', '#e74c3c', '#2ecc71']
plt.pie(fenomeno_counts.values, labels=fenomeno_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=colors_fenomeno, textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Distribuição de Casos por Fenômeno Climático', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('static/images/fenomeno_climatico_distribuicao.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: static/images/fenomeno_climatico_distribuicao.png")
plt.close()

# 7. Distribuição por Intensidade do Fenômeno
print("7. Gerando gráfico de distribuição por intensidade do fenômeno...")
plt.figure(figsize=(12, 6))
intensidade_counts = df['INTENS_FENOM'].value_counts()
plt.barh(intensidade_counts.index, intensidade_counts.values, color='orange', edgecolor='black')
plt.xlabel('Número de Casos', fontsize=12, fontweight='bold')
plt.ylabel('Intensidade', fontsize=12, fontweight='bold')
plt.title('Distribuição de Casos por Intensidade do Fenômeno Climático', 
          fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/intensidade_fenomeno_distribuicao.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: static/images/intensidade_fenomeno_distribuicao.png")
plt.close()

# 8. Distribuição de Hospitalização
print("8. Gerando gráfico de distribuição de hospitalização...")
plt.figure(figsize=(10, 6))
hosp_counts = df['HOSPITALIZ'].value_counts()
colors_hosp = ['#2ecc71', '#e74c3c', '#95a5a6']
plt.pie(hosp_counts.values, labels=hosp_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=colors_hosp, textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Distribuição de Casos por Hospitalização', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('static/images/hospitalizacao_distribuicao.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: static/images/hospitalizacao_distribuicao.png")
plt.close()

print("\n" + "="*80)
print("TODOS OS GRÁFICOS FORAM GERADOS COM SUCESSO!")
print("="*80)
print(f"\nTotal de casos analisados: {len(df):,}")
print(f"Período: {df['NU_ANO'].min()} - {df['NU_ANO'].max()}")
print(f"Casos hospitalizados: {len(df[df['HOSPITALIZ'] == 'SIM']):,} ({len(df[df['HOSPITALIZ'] == 'SIM'])/len(df)*100:.2f}%)")
print(f"Idade média: {df['IDADE'].mean():.1f} anos")
print("\nArquivos gerados em: static/images/")
