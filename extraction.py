import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
import ast
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


"""
EXTRACCION
"""
path_files = r'C:\Users\Daniel\Desktop\Universidad\2020-2\Datawarehouse\2daEvaluacion\2da'
df_credits = pd.read_csv(path_files+'\credits.csv')
df_meta = pd.read_csv(path_files+'\movies_metadata.csv')

# SE CREA UNA COLUMNA DE AÑO APARTIR DE LA COLUMNA RELEASE_DATE
df_meta['release_date'] = pd.to_datetime(df_meta['release_date'],errors='coerce')
df_meta['year'] = df_meta['release_date'].dt.year

# LIMPIANDO LA COLUMNA ID EN DF_META (REMPLAZO - a 0) Y CASTEO A INT LA COLUMNA
df_meta['id'] = df_meta['id'].apply(lambda x: x.replace('-','0'))

df_meta['id'] = df_meta['id'].astype(int)

# FUSIONO LA DATA DE CREDITS Y METADATA
merged_data = pd.merge(df_meta,df_credits,on='id')

# VEO COMO QUEDA LA DATA
#print(merged_data.sample(1))

"""
TRANSFORMACION Y CARGA
"""

# SE ELIMINAN LAS COLUMNAS QUE NO UTILIZAREMOS PARA EL EJEMPLO

data_to_use = merged_data.drop(['belongs_to_collection','homepage','homepage','poster_path','tagline','video','adult','imdb_id','status'],axis=1)

# SE BUSCA LOS VALORES NULOS EN ESTE NUEVO DATA FRAME CREADO

"""

plt.figure(figsize=(8,5))
sns.heatmap(data_to_use.isnull(),cmap='Blues',cbar=False,yticklabels=False)
plt.show()

"""

# AST RETORNA LAS CARACTRISTICAS MÁS IMPORTANTES DE LAS COLUMNAS GENRES, CAST, CREW Y LAS MUESTRA DE FORMA LEGIBLE

data_to_use['genres'] = data_to_use['genres'].map(lambda x: ast.literal_eval(x)) # 100 -> 90 (CORRELACION, PROXIMIDAD)
data_to_use['cast'] = data_to_use['cast'].map(lambda x: ast.literal_eval(x))
data_to_use['crew'] = data_to_use['crew'].map(lambda x: ast.literal_eval(x))

# SE REEMPLAZA LOS VALORES NAN POR unknown Y SE LIMPIA 
print("==============================================================")
#print("=======================REPLACE================================")
data_to_use['production_companies'] = data_to_use['production_companies'].replace(np.nan,'unknown')
data_to_use['production_countries'] = data_to_use['production_countries'].replace(np.nan,'unknown')
data_to_use['spoken_languages'] = data_to_use['spoken_languages'].replace(np.nan,'unknown')

# {'name': 'Pixar Animation Studios', 'id': 3} ==> 'name': 'Pixar Animation Studios'
data_to_use['production_company_name_only_first'] = data_to_use['production_companies'].apply(lambda x: x.split(',')[0]) # ""
# 'name': 'Pixar Animation Studios' ==> Pixar Animation Studios
data_to_use['production_company_name_only_first'] = data_to_use['production_company_name_only_first'].apply(lambda x: x.split(':')[-1])
#print("=======================SPOKEN================================")
#print()
data_to_use.spoken_languages[1112]
data_to_use['spoken_languages_only'] = data_to_use['spoken_languages'].apply(lambda x: x.split(',')[-1])
data_to_use['spoken_languages_only1'] = data_to_use['spoken_languages'].apply(lambda x: x.split(':')[-1])
data_to_use['spoken_languages_only12'] = data_to_use['spoken_languages_only1'].apply(lambda x : x[:-2])
#print(data_to_use.head(1))



def make_genresList(x):
    gen = []
    st = " "
    for i in x:
        if i.get('name') == 'Science Fiction':
            scifi = 'Sci-Fi'
            gen.append(scifi)
        else:
            gen.append(i.get('name'))
    if gen == []:
        return np.NaN
    else:
        return (st.join(gen))
# [{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]}


# [ANIMATION,COMEDY] => "ANIMATION COMEDY"
data_to_use['genres_list'] = data_to_use['genres'].map(lambda x: make_genresList(x))


# MUESTRA CARACTERISTICAS COMO NOMBRE DE ACTOR EN FORMATO DE LECTURA

def get_actor1(x):
    casts = []
    for i in x:
        casts.append(i.get('name'))
    if casts == []:
        return np.NaN
    else:
        return (casts[0])

data_to_use['actor_1_name'] = data_to_use['cast'].map(lambda x: get_actor1(x))

def get_actor2(x):
    casts = []
    for i in x:
        casts.append(i.get('name'))
    if casts == [] or len(casts)<=1:
        return np.NaN
    else:
        return (casts[1])

data_to_use['actor_2_name'] = data_to_use['cast'].map(lambda x: get_actor2(x))

def get_actor3(x):
    casts = []
    for i in x:
        casts.append(i.get('name'))
    if casts == [] or len(casts)<=2:
        return np.NaN
    else:
        return (casts[2])

data_to_use['actor_3_name'] = data_to_use['cast'].map(lambda x: get_actor3(x))

def get_directors(x):
    dt = []
    st = " "
    for i in x:
        if i.get('job') == 'Director':
            dt.append(i.get('name'))
    if dt == []:
        return np.NaN
    else:
        return (st.join(dt))

data_to_use['director_name'] = data_to_use['crew'].map(lambda x: get_directors(x))

"""
LOAD
"""
data_to_use.to_excel("data_para_usar.xlsx")


"""
PREDICCION
"""
print("========================================")
#print(data_to_use.head(1))
data_for_analyses = data_to_use.drop(['genres','id','original_language','production_companies','release_date','spoken_languages','cast','crew','spoken_languages_only','spoken_languages_only1'],axis=1)
# CONVIRTIENDO LA COLUMNA BUDGET A INT
data_for_analyses['budget'] = data_for_analyses['budget'].astype(int)

data_for_analyses.describe()

# EL VALOR MAS ALTO EN EL REVENUE ES 2787965087.0
# SE MUESTRA LA PELICULA CON EL MEJOR REVENUE(INGRESO)
data_for_analyses[data_for_analyses['revenue']==2787965087.0] # AVATAR
# SE CREA UNA NUEVA COLUMNA LLAMADA GANANCIAS = VENTAS - GASTOS
data_for_analyses['Profit'] = data_for_analyses['revenue'] - data_for_analyses['budget']

# SE HACE UN GRAFICO PARA MOSTRAR POR AÑOS LA CANTIDAD DE PELICULAS ESTRENEDAS
data_for_analyses.year.value_counts(dropna=False).sort_index().plot(kind='barh',color='g',figsize=(20,40))
data_for_analyses['popularity'] = data_for_analyses['popularity'].astype(float)
data_for_analyses_clean = data_for_analyses.dropna(how='any')
plt.figure(figsize=(8,5))
sns.heatmap(data_for_analyses_clean.isnull(),cmap='Blues',cbar=False,yticklabels=False)

# GENERA UN GRAFICO PARA SABER LA CORRELACION DE LAS COLUMNA
correlations = data_for_analyses_clean.corr()
f,ax = plt.subplots(figsize=(10,6))
sns.heatmap(correlations, annot=True, cmap="YlGnBu", linewidths=.5)

# SE CREA UN NUEVO DATA FRAME PARA APLICAR ALGORITMOS DE ML
df_for_ML = data_for_analyses_clean[['budget','popularity','revenue','runtime','vote_average','vote_count','year','Profit']]
df_for_ML.hist(bins=30,figsize=(15,8),color='g')

for i in df_for_ML.columns:
    axis = data_for_analyses_clean.groupby('vote_average')[[i]].mean().plot(figsize=(10,5),marker='o',color='g')

plt.show()

X = df_for_ML.drop('Profit',axis=1)
y = df_for_ML['Profit']
print(X.shape,y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lm = LinearRegression()
lm.fit(X_train,y_train)

prec_lm=lm.predict(X_test)

print('El error cuadratico medio utilizando regresion es: ',mean_squared_error(y_test,prec_lm))
print('El error absoluto medio utilizando regresion es: ',mean_absolute_error(y_test,prec_lm))

# Random forest es uno de los algoritmos de prediccion más utilizado a nivel de competencias, dando resultados muy rapidos en comparación a una regresión lineal
rf = RandomForestRegressor()
rf.fit(X_train,y_train)

prec_rf=rf.predict(X_test)

print('El error cuadratico medio utilizando random forest es: ',mean_squared_error(y_test,prec_rf))
print('El error absoluto medio utilizando random forest es: ',mean_absolute_error(y_test,prec_rf))