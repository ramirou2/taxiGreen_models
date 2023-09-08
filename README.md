# MODELOS ASOCIADOS AL PROYECTO TAXI GREEN 
### Se presentan los modelos asociados a la entrega del proyecto Taxi Green, en este caso se realizó la construcción de dos modelos. Uno que corresponde a las influencias de las variables climáticas en las características de una hora de viajes en Manhattan y el otro realizá una optimización de la flota en términos de inversión y cantidad de vehículos, priorizando esta última variable.

### [Repositorio Proyecto General](https://github.com/ramirou2/NY_TaxiGreen)

## Modelo 1: Modelo ML - Clima Descripción General
### En este modelo se empleo *Machine Learning*, puntualmente un modelo de *Random Forest*, el cual fue entrenado con datos de los viajes de los *Yellow Taxis* en Manhattan - New York asociado a los datos climáticos correspondientes al momento del viaje.
- Variables climáticas: Temperatura, Nieve, Lluvia
- Variables temporales: Hora, Día de la semana.
- Variables de viajes: Cantidad de viajes por hora, Total de km recorridos, Total de dinero recaudado

### Stack tecnológico 
- Python: Pandas, Numpy, Matplotlib, Seaborn, Scikit Learn, Request, Os.

### Scripts asociados
- [Modelos](./modelos_ml.py)
- [Exploración inicial](./EDA_models.ipynb)

## Modelo 2: Modelo Optimización de flota
### En términos de las condiciones propuestas, 60% de flota eléctrica, minimización de emisiones y minimización de ruidos. Se realizó un modelo de optimización empleando metodologías lineales, que si bien parecen simplificado tienen un muy buen desempeño y a gran velocidad sin necesidad de recurrir a metodologías más complejas.
- Entrada: Datos de autos autorizados para desempeño de viajes, Cantidad de autos necesarios, Dinero inicial
- Salida: Flota optimizada en términos de la entrada y las restricciones asociadas al proyecto.

### Stack tecnológico
- Python: Os, Pulp, Pandas, Numpy

### Scripts asociados
- [Modelos](./modelos_ml.py)
- [Exploración inicial](./EDA_models.ipynb)

## Despliegue de los modelos
### En términos de conseguir un despliegue de bajo o nulo costo se analizaron diversas alternativas, quedando por encima del resto **Streamlit Cloud**, por ende se utilizó su *framework* asociado para desarrollar un despliegue MVP: Producto Viable Mínimo.

### Stack tecnológico
- Python: Streamlit, Pandas, Numpy, Request, Seaborn, Matplotlib
- Despliegue: Streamlit Cloud

### Scripts asociados
- [Aplicación - Streamlit](./app.py)

### Link del modelo --> [ModeloOnline](https://taxigreenmodels-pf.streamlit.app)

## Sección de Vista Previa
![Captura de Pantalla 1](/src/screenshot1.png)
![Captura de Pantalla 2](/src/screenshot2.png)
