import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import numpy as np
import modelos_ml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ.get('PYTHON_VERSION')

#Función para nuestra animación
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    else:
        return r.json()

lottie_taxi =   load_lottieurl("https://lottie.host/e51b6be4-1ea4-4b20-a34a-1820a759d8f4/qq9JLCQUKn.json")
lottie_taxi_2 = load_lottieurl("https://lottie.host/ddb625f8-24f8-4e93-b3e8-dabd7c88d6b6/ZfHotMJFrV.json")

dict_days =  {
    'Lunes': 0,
    'Martes': 1,
    'Miércoles': 2,
    'Jueves': 3,
    'Viernes': 4,
    'Sábado': 5,
    'Domingo': 6
}

@st.cache(allow_output_mutation=True)
def cargar_modelo():
    return modelos_ml.complete_exec_2()

# Inicializa el espacio de estado de la sesión
if 'flota_obtenida' not in st.session_state:
    st.session_state.flota_obtenida = pd.DataFrame()

# Llama a la función para cargar el modelo una vez al inicio de la aplicación
model, scaler = cargar_modelo()

with st.container():
    st.subheader("LBD - DataTeam")
    st.write("---")
    st.title("Modelos del proyecto Taxi Green")
    st.write("Se presentan dos modelos en formato de MVP, entorno y datos reducidos")

    st.markdown('<a href="https://github.com/ramirou2/NY_TaxiGreen"><img src="https://cdn-icons-png.flaticon.com/256/25/25231.png" width="50"></a>', unsafe_allow_html=True)

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Modelo 1 - Clima ML")
        st.write("""
        En este modelo se plantea un adelanto simplificado de un modelo Random Forest.Para calcular cuantos taxis aproximados se necesitan según el clima, día y hora.
            """
            )
        st.write("[Repositorio Modelo](https://youtube.com)")
    
    with right_column:
        st_lottie(lottie_taxi, height=300, key="taxi1")

    st.write("")



    st.header("Ingrese los datos:")
    dia = st.selectbox("Día:", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])
    hora = st.number_input("Hora:", min_value=0, step=1, max_value=23)
    temperatura = st.number_input("Temperatura en °C:", min_value=0, max_value=50)
    lluvia = st.number_input("Lluvia en mm:")
    nieve = st.number_input("Nieve en cm:")

    # Espacio en blanco para mover el botón hacia abajo
    st.write("")  # Esto crea un espacio en blanco

    # Botón para enviar los datos al backend
    if st.button("Enviar Datos"):
        # Crear un diccionario con los datos
        datos = {
            'precipitation': float(lluvia),
            'snowfall': float(nieve),
            'apparent_temperature': float(temperatura),
            'day_of_week': int(dict_days[str(dia)]),
            'hour': int(hora)
        }

        # Convertir los datos de entrada en una matriz bidimensional
        datos = np.array([[datos['precipitation'], datos['snowfall'], datos['apparent_temperature'], datos['day_of_week'], datos['hour']]])
        datos = scaler.transform(datos)
        # Realizar la predicción con el modelo

        salida = st.columns(4)
        with salida[0]:
            st.write("Salida en cantidad de viajes:")
            st.write( int(model.predict(datos)) )
        with salida[1]:
            st.write("Salida en cantidad de taxis promedio:")
            st.write( int(model.predict(datos)/3) )
        with salida[2]:
            st.write("Salida en cantidad de km recorridos:")
            st.write( round(float(model.predict(datos) * 1.5), 2) )

with st.container():
    st.write("---")
    st.header("Modelo 2 - Flota según inversión")
    st.write("""En este modelo se ingresa una cantidad inicial de inversión y la cantidad de autos que se esperan.
                    Se prioriza la cantidad de autos pero se tiene en cuenta el monto ingresado para que sea lo minimo posible.
                Se respeta el 60\% de flota inicial eléctrica """)
    
    st_lottie(lottie_taxi_2, height=300, key="taxi2")

    st.header("Ingrese los datos:")
    inversion_inicial = st.number_input("Inversión:", min_value=20000)
    n_autos = st.number_input("Tamaño de flota:", min_value=1, step=1)

    
    if st.button("Obtener Flota"):
        flota_obtenida, inversion_inicial = modelos_ml.buscar_inversion(int(inversion_inicial), int(n_autos))
        flota_obtenida = flota_obtenida[['Model', 'Manufacturer', 'fuelType1','co2', 'Noise', 'fuelCost08', 'range']]
        st.write(flota_obtenida[['Model', 'Manufacturer', 'fuelType1']])
        st.write(inversion_inicial)
        st.session_state.flota_obtenida = flota_obtenida
    
    # Sección gráficos de apoyo visual
    if st.button("Simular flota en acción"):
        flota_obtenida = st.session_state.flota_obtenida
        st.title('Gráficos asociados a gasto en combustibles')
        # Preparación de autos ejemplo
        autos = pd.read_csv('./data/autos_aprobados.csv').sort_values(by = ['Price'])
        electrico = autos[autos['fuelType1'] == 'Electricity'].iloc[0]
        regular = autos[autos['fuelType1'] == 'Regular Gasoline'].iloc[0]
        autos = []
        autos.append(electrico)
        autos.append(regular)
        autos = pd.DataFrame(autos)
        autos['fuelCostPerKm'] = autos['fuelCost08'] / autos['range'] 
        autos['fuelCostPerYear'] = autos['fuelCostPerKm'] * 300 * 365
        autos['modelComplete'] = autos['Manufacturer'] + autos['Model']
        autos['co2PerYear'] = autos['co2'] * 300 * 365
        
        
        #Primer gráfico ______________________________________________________________________
        st.subheader('Gasto en Combustible en 1 año según automóvil')
        st.write('Tomando como ejemplo un auto electrico y un auto a combustible regular se compara el consumo en un año, a 300 km por día')

        sns.set_style("darkgrid")
        fig = plt.figure(figsize=(10, 4))
        ax = sns.barplot(data=autos, x='modelComplete', y='fuelCostPerYear')
        plt.xlabel('Automóvil')
        plt.ylabel('Gasto en Combustible')
        plt.tight_layout()
        
        ax.set_facecolor('none')  # Fondo transparente
        fig.patch.set_facecolor('none')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        st.pyplot(fig)

        #Segundo gráfico, según flota obtenida y una flota regular________________________________________________
        st.subheader('Gasto en Combustible en 1 año según flota obtenida Vs flota regular')
        st.write('Se compara el consumo en un año, a 300 km por día de una flota regular vs la obtenida')
        n_electricos = flota_obtenida[flota_obtenida['fuelType1'] == 'Electricity'].shape[0]
        n_regular = flota_obtenida[flota_obtenida['fuelType1'] == 'Regular Gasoline'].shape[0]
        st.markdown(f'La flota esta compuesta por {n_electricos} autos electricos y {n_regular} a gasolina regular')

        # Gasto combustible
        flota_obtenida['fuelCostPerKm'] = flota_obtenida['fuelCost08'] / flota_obtenida['range'] 
        flota_obtenida['fuelCostPerYear'] = flota_obtenida['fuelCostPerKm'] * 300 * 365

        flotaComb = flota_obtenida[flota_obtenida['fuelType1'] == 'Electricity']['fuelCostPerYear'].sum() + flota_obtenida[flota_obtenida['fuelType1'] == 'Regular Gasoline']['fuelCostPerYear'].sum()
        flotaRegular = (n_electricos + n_regular) * autos[autos['fuelType1'] == 'Regular Gasoline']['fuelCostPerYear'].iloc[0]

        dataFuel = pd.DataFrame( {'Tipo de Flota': ['flota', 'flota Regular'],
                                    'Gasto Anual Total': [flotaComb, flotaRegular] })
        
        sns.set_style("darkgrid")
        fig2 = plt.figure(figsize=(10, 4))
        ax = sns.barplot(x='Tipo de Flota', y='Gasto Anual Total', data=dataFuel, palette="viridis")
        ax.set_facecolor('none')  # Fondo transparente
        fig2.patch.set_facecolor('none')
        plt.xlabel('Flota Vs Flota Regular', color='white')
        plt.ylabel('Gasto Anual Total', color='white')
        # Personaliza los colores de las líneas de los ejes
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        st.pyplot(fig2)

        # tercer grafico, Emisiones ________________________________________________________________________________________-
        # Emisiones de Co2
        flota_obtenida['co2PerYear'] = flota_obtenida['co2'] * 300 * 365
        st.subheader('Emisiones de Co2 un año flota obtenida Vs flota regular')
        st.write('Se comparan las emisiones de la flota obtenida vs una regular')
        st.markdown(f'La flota esta compuesta por {n_electricos} autos electricos y {n_regular} a gasolina regular')

        flota = flota_obtenida[flota_obtenida['fuelType1'] == 'Electricity']['co2PerYear'].sum() + flota_obtenida[flota_obtenida['fuelType1'] == 'Regular Gasoline']['co2PerYear'].sum()
        flotaRegular = (n_electricos + n_regular) * autos[autos['fuelType1'] == 'Regular Gasoline']['co2PerYear'].iloc[0]

        dataFuel = pd.DataFrame( {'Tipo de Flota': ['flota', 'flota Regular'],
                                    'Emisiones Totales': [flota, flotaRegular] })
        
        sns.set_style("darkgrid")
        fig3 = plt.figure(figsize=(10, 4))
        ax = sns.barplot(x='Tipo de Flota', y='Emisiones Totales', data=dataFuel, palette="viridis")
        ax.set_facecolor('none')  # Fondo transparente
        fig3.patch.set_facecolor('none')
        plt.xlabel('Flota Vs Flota Regular', color='white')
        plt.ylabel('Emisiones Totales', color='white')
        # Personaliza los colores de las líneas de los ejes
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        st.pyplot(fig3)
    st.write("[Repositorio Modelo](https://youtube.com)")