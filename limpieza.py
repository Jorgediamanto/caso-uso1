import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Cargar los datos
data = pd.read_csv('cars.csv', sep=';')

# 1. LIMPIEZA DE DATOS

# Reemplazar valores "NO DATA" o similares con NaN
data.replace(["NO DATA", "N/A", "n/a", "?", "", " "], np.nan, inplace=True)

# Rellenar valores faltantes (media para numéricos, moda para categóricos)
for column in data.columns:
    if data[column].dtype == 'object':  # variables categóricas
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:  # variables numéricas
        data[column].fillna(data[column].mean(), inplace=True)

# 1.2 Eliminación de outliers usando el rango intercuartil (IQR)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Aplicamos el tratamiento de outliers solo a las columnas numéricas relevantes
numeric_cols = ['EDAD_COCHE', 'COSTE_VENTA', 'km_anno', 'Edad Cliente', 'Tiempo']
for col in numeric_cols:
    data = remove_outliers(data, col)

# 2. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# Usamos LabelEncoder para convertir variables categóricas en formato numérico
categorical_cols = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Guardar el codificador para su uso futuro

# 3. NORMALIZACIÓN Y ESCALADO
# Escalado de las variables numéricas
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Guardar el archivo ya procesado en un nuevo archivo CSV limpio
data.to_csv('carsclean.csv', index=False)

print("Archivo limpio guardado como 'ruta_a_tu_archivo_limpio.csv'")
