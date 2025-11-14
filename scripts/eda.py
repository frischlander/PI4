
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv('df_dengue_tratado.csv')

# --- Análise Inicial ---
print('Informações do DataFrame:')
df.info()
print('\nPrimeiras 5 linhas do DataFrame:')
print(df.head())
print('\nEstatísticas Descritivas:')
print(df.describe(include='all'))

# Converter colunas de data para o tipo datetime
df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'])
df['DT_SIN_PRI'] = pd.to_datetime(df['DT_SIN_PRI'])

# --- EDA para variáveis categóricas ---

# Distribuição de Sexo
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='CS_SEXO')
plt.title('Distribuição de Casos por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Número de Casos')
plt.savefig('sexo_distribuicao.png')
plt.close()

# Distribuição de Raça
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='CS_RACA', order=df['CS_RACA'].value_counts().index)
plt.title('Distribuição de Casos por Raça')
plt.xlabel('Raça')
plt.ylabel('Número de Casos')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('raca_distribuicao.png')
plt.close()

# Distribuição de Hospitalização
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='HOSPITALIZ')
plt.title('Distribuição de Casos por Hospitalização')
plt.xlabel('Hospitalização')
plt.ylabel('Número de Casos')
plt.savefig('hospitalizacao_distribuicao.png')
plt.close()

# Distribuição de Fenômeno Climático
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='FENOMENO', order=df['FENOMENO'].value_counts().index)
plt.title('Distribuição de Casos por Fenômeno Climático')
plt.xlabel('Fenômeno Climático')
plt.ylabel('Número de Casos')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fenomeno_climatico_distribuicao.png')
plt.close()

# Distribuição de Intensidade do Fenômeno Climático
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='INTENS_FENOM', order=df['INTENS_FENOM'].value_counts().index)
plt.title('Distribuição de Casos por Intensidade do Fenômeno Climático')
plt.xlabel('Intensidade do Fenômeno')
plt.ylabel('Número de Casos')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('intensidade_fenomeno_distribuicao.png')
plt.close()

# --- EDA para variáveis numéricas ---

# Distribuição de Idade
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='IDADE', bins=30, kde=True)
plt.title('Distribuição de Idade dos Casos de Dengue')
plt.xlabel('Idade')
plt.ylabel('Número de Casos')
plt.savefig('idade_distribuicao.png')
plt.close()

# --- Análise temporal ---

# Casos de dengue ao longo do tempo (por ano)
casos_por_ano = df['DT_NOTIFIC'].dt.year.value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=casos_por_ano.index, y=casos_por_ano.values)
plt.title('Número de Casos de Dengue por Ano')
plt.xlabel('Ano')
plt.ylabel('Número de Casos')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('casos_por_ano.png')
plt.close()

# Casos de dengue ao longo do tempo (por mês)
df['MES_NOTIFIC'] = df['DT_NOTIFIC'].dt.month
casos_por_mes = df.groupby(['NU_ANO', 'MES_NOTIFIC']).size().reset_index(name='contagem')
plt.figure(figsize=(12, 7))
sns.lineplot(data=casos_por_mes, x='MES_NOTIFIC', y='contagem', hue='NU_ANO', palette='viridis')
plt.title('Número de Casos de Dengue por Mês e Ano')
plt.xlabel('Mês')
plt.ylabel('Número de Casos')
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
plt.grid(True)
plt.tight_layout()
plt.savefig('casos_por_mes_ano.png')
plt.close()

print('\nEDA concluída e gráficos salvos como arquivos PNG.')

