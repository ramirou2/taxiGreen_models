import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import os
import requests
import sys

import pulp

#_______________________MODELO DE OPTIMIZACIÓN DE FLOTAS_________________________________________
def optimizar_flota(n_autos, autos, inversion):
    # Separar autos eléctricos y de gasolina
    autos_electricos = autos[autos['fuelType1'] == 'Electricity']
    autos_gasolina = autos[autos['fuelType1'] == 'Regular Gasoline']
    
    # Crear un problema de programación lineal
    problema = pulp.LpProblem("Optimizacion de Flota", pulp.LpMinimize)
    
    # Variables de decisión: Cantidad de cada auto a comprar
    cantidad_autos_electricos = pulp.LpVariable.dicts("CantidadAutosElectricos", autos_electricos.index, lowBound=0, cat='Integer')
    cantidad_autos_gasolina = pulp.LpVariable.dicts("CantidadAutosGasolina", autos_gasolina.index, lowBound=0, cat='Integer')
    
    # Función objetivo: Minimizar el ruido y las emisiones
    problema += pulp.lpSum([autos_electricos.loc[i, 'Noise'] * cantidad_autos_electricos[i] for i in autos_electricos.index]) + \
                pulp.lpSum([autos_gasolina.loc[i, 'Noise'] * cantidad_autos_gasolina[i] for i in autos_gasolina.index]) + \
                pulp.lpSum([autos_electricos.loc[i, 'co2'] * cantidad_autos_electricos[i] for i in autos_electricos.index]) + \
                pulp.lpSum([autos_gasolina.loc[i, 'co2'] * cantidad_autos_gasolina[i] for i in autos_gasolina.index])
    
    # Restricción: No superar el monto de inversión
    problema += pulp.lpSum([autos_electricos.loc[i, 'Price'] * cantidad_autos_electricos[i] for i in autos_electricos.index]) + \
                pulp.lpSum([autos_gasolina.loc[i, 'Price'] * cantidad_autos_gasolina[i] for i in autos_gasolina.index]) <= inversion
    
    # Restricción: Obtener al menos el 60% de autos eléctricos
    min_autos_electricos = int(0.6 * n_autos)
    problema += pulp.lpSum([cantidad_autos_electricos[i] for i in autos_electricos.index]) >= min_autos_electricos

    # Restricción: Obtener exactamente n_autos en total
    problema += pulp.lpSum([cantidad_autos_electricos[i] for i in autos_electricos.index]) + \
                pulp.lpSum([cantidad_autos_gasolina[i] for i in autos_gasolina.index]) == n_autos
    
    # Resolver el problema de optimización
    problema.solve()
    
    if pulp.LpStatus[problema.status] == 'Optimal':
        # Construir la lista de autos seleccionados y sus cantidades
        autos_seleccionados_electricos = [(i, int(cantidad_autos_electricos[i].value())) for i in autos_electricos.index if cantidad_autos_electricos[i].value() > 0]
        autos_seleccionados_gasolina = [(i, int(cantidad_autos_gasolina[i].value())) for i in autos_gasolina.index if cantidad_autos_gasolina[i].value() > 0]
        
        return autos_seleccionados_electricos + autos_seleccionados_gasolina
    else:
        return [(0,0)]

def decodificar_flota(autos_seleccionados, autos):
    # Crear una lista para almacenar temporalmente las filas seleccionadas
    autos_seleccionados_lista = []
    
    # Iterar sobre las tuplas de autos seleccionados
    for auto_idx, cantidad in autos_seleccionados:
        # Obtener el auto del DataFrame original por su índice
        auto = autos.loc[auto_idx]
        
        # Repetir el auto la cantidad especificada y agregarlo a la lista
        for i in range(cantidad):
            autos_seleccionados_lista.append(auto)
    
    # Crear un DataFrame a partir de la lista de filas seleccionadas
    autos_seleccionados_df = pd.DataFrame(autos_seleccionados_lista)
    
    return autos_seleccionados_df

def buscar_inversion(inversion_inicial, n_autos_fijos):
    df_car = pd.read_csv('./data/autos_aprobados.csv')
    incremento = 10000
    autosFlota = pd.DataFrame()
    while True:
        autos_seleccionados = optimizar_flota(n_autos_fijos, df_car, inversion_inicial)
        cantidad_autos_seleccionados = sum([cantidad[1] for cantidad in autos_seleccionados])
        
        if cantidad_autos_seleccionados == n_autos_fijos:
            autosFlota = decodificar_flota(autos_seleccionados, df_car)
            inversion_inicial = autosFlota['Price'].sum()
            return autosFlota, inversion_inicial
            break  # Hemos alcanzado la cantidad fija de autos
        else:
            inversion_inicial += incremento  # Aumentar el presupuesto en caso contrario


# _______________________________________________________________________________________________
# _______________________________________________________________________________________________
#_______________________MODELO DE CONDICIONES CLIMÁTICAS_________________________________________

def ml_clima(data):
    X = data[['precipitation', 'snowfall', 'apparent_temperature', 'day_of_week', 'hour']]
    y = data[['cantidad_viajes']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definir los hiperparámetros a ajustar y sus rangos
    param_grid = {
        'n_estimators': [50, 100, 150,200],  # Diferentes números de árboles
        'max_depth': [None, 10, 20, 50],     # Diferentes profundidades máximas
        'min_samples_split': [2, 5, 10, 20] # Diferentes valores mínimos para dividir un nodo
    }

    # Crear el modelo de Regresión de Bosque Aleatorio
    rf_model = RandomForestRegressor(random_state=42) 

    # Realizar la búsqueda de hiperparámetros con validación cruzada
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)

    # Obtener los mejores hiperparámetros
    best_params = grid_search.best_params_

    # Entrenar el modelo final con los mejores hiperparámetros
    best_rf_model = RandomForestRegressor(random_state=42, **best_params)
    best_rf_model.fit(X_train_scaled, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = best_rf_model.predict(X_test_scaled)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return (best_rf_model, scaler,mse, r2, best_params)

def wea_data(start_date, end_date):
    base_url = 'https://archive-api.open-meteo.com/v1/archive?'
    params = {
        'latitude'  : '40.7834',
        'longitude' : '-73.9663',
        'start_date': start_date,
        'end_date'  : end_date,
        'hourly'     : 'apparent_temperature,precipitation,rain,snowfall',
        'timezone' :  'auto'
    }

    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        return f'Error al realizar la solicitud:{response.status_code}'
    else:
        data_json = response.json()
        df_clima = pd.DataFrame(data_json["hourly"])
        df_clima['time'] = pd.to_datetime(df_clima['time'])
        return df_clima

def prep_data(clima_data, path_taxi, zones, start_date, end_date):
    files = os.listdir(path_taxi)
    data = pd.DataFrame()
    for file in files:
        df = pd.read_parquet(path_taxi + file)
        df.columns = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'airport_fee']
        df.drop(columns=['VendorID','RatecodeID', 'store_and_fwd_flag', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',  'airport_fee' ], inplace=True)
        
        #Filtramos por zona, MVP va con manhattan y en tiempo de análisis
        df = df[df['PULocationID'].isin(zones)]
        filtro = (df['tpep_pickup_datetime'] >= start_date) & (df['tpep_pickup_datetime'] <= end_date)
        df = df[filtro]

        # Completamos los datos con un taxi representativo
        standardTaxi = 9999
        co2 = 290.63
        df['vehicle_id'] = [standardTaxi] * len(df)
        df['co2_fingerprint'] = [0] * len(df)
        df['co2_fingerprint'] = df['co2_fingerprint'].apply(lambda x: co2)

        #Concatenamos la data de taxis
        data = pd.concat([data, df], axis = 0)
    
    # Se realiza el merge entre los datos de viaje y los datos de clima en terminos del comienzo del viaje
    clima_data['time'] = clima_data['time'].astype('<M8[us]') #Genera error sino, en script, en notebook  no.
    merged = merged = pd.merge_asof(data.sort_values('tpep_pickup_datetime'), clima_data.sort_values('time'), right_on='time', left_on='tpep_pickup_datetime', direction='nearest')
    del data
    del df
    merged = merged.groupby(['time']).agg({
        'precipitation': 'first',
        'snowfall': 'first',
        'apparent_temperature': 'first',
        'trip_distance': 'sum' }).join( merged.groupby(['time']).size().reset_index(name='cantidad_viajes').set_index(['time']))
    
    merged['time'] = merged.index
    merged['day_of_week'] = merged['time'].dt.dayofweek  # 0=lunes, 1=martes, ..., 6=domingo
    merged['hour'] = merged['time'].dt.hour
    merged = merged.reset_index(drop=True)

    return merged

def prep_data_red(clima_data, data, start_date, end_date):

    filtro = (data['time'] >= start_date) & (data['time'] <= end_date)
    data = data[filtro]
    merged_data = pd.merge_asof(data.sort_values('time'), clima_data.sort_values('time'), right_on='time', left_on='time', direction='nearest')
    return merged_data

def complete_exec_2():
    start_date = '2023-02-01'
    end_date = '2023-05-31'

    zones = pd.read_csv('./data/taxi+_zone_lookup.csv')
    manhattan = zones[ zones['Borough'] == 'Manhattan']
    zones_man = manhattan['LocationID'].to_list()
    #Libero memoria pensand en render
    del manhattan
    del zones

    clima_data = wea_data(start_date, end_date)
    data = pd.read_csv('./data/taxi/data_prep.csv')
    data['time'] = pd.to_datetime(data['time'])
    data = prep_data_red(clima_data, data, start_date, end_date)

    #Libero memoria pensando en render
    del clima_data

    #Acá se hace eso para disminuir recursos.
    X = data[['precipitation', 'snowfall', 'apparent_temperature', 'day_of_week', 'hour']]
    y = data[['cantidad_viajes']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Entrenar el modelo final con los mejores hiperparámetros
    best_params = {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 50}
    best_rf_model = RandomForestRegressor(random_state=42, **best_params)
    best_rf_model.fit(X_train_scaled, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = best_rf_model.predict(X_test_scaled)
    return best_rf_model, scaler

def complete_exec():
    start_date = '2023-02-01'
    end_date = '2023-05-31'

    zones = pd.read_csv('./data/taxi+_zone_lookup.csv')
    manhattan = zones[ zones['Borough'] == 'Manhattan']
    zones_man = manhattan['LocationID'].to_list()
    #Libero memoria pensand en render
    del manhattan
    del zones

    clima_data = wea_data(start_date, end_date)
    data = prep_data(clima_data, './data/taxi/', zones_man, start_date, end_date)

    #Libero memoria pensando en render
    del clima_data

    #Acá se hace eso para disminuir recursos.
    X = data[['precipitation', 'snowfall', 'apparent_temperature', 'day_of_week', 'hour']]
    y = data[['cantidad_viajes']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Entrenar el modelo final con los mejores hiperparámetros
    best_params = {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 200}
    best_rf_model = RandomForestRegressor(random_state=42, **best_params)
    best_rf_model.fit(X_train_scaled, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = best_rf_model.predict(X_test_scaled)
    return best_rf_model
        
if __name__ == "__main__":
    start_date = '2023-02-01'
    end_date = '2023-05-31'

    zones = pd.read_csv('./data/taxi+_zone_lookup.csv')
    manhattan = zones[ zones['Borough'] == 'Manhattan']
    zones_man = manhattan['LocationID'].to_list()
    #Libero memoria pensand en render
    del manhattan
    del zones

    clima_data = wea_data(start_date, end_date)
    #data = prep_data(clima_data, './data/taxi/', zones_man, start_date, end_date)
    data = pd.read_csv('./data/taxi/data_prep.csv')
    data['time'] = pd.to_datetime(data['time'])
    data = prep_data_red(clima_data, data, start_date, end_date)
    #Libero memoria pensando en render
    del clima_data

    model, scaler, mse, r2, best_params = ml_clima(data)

    print(f'Mean Squared Error (Random Forest): {mse}')
    print(f'R-squared (Random Forest): {r2}')
    print(f'Best Hyperparameters: {best_params}') #  {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 50}
    print(f'Tamaño del modelo { (sys.getsizeof(model)) / (1024 * 1024)}')
