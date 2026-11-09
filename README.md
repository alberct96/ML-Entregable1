# Práctica 1 ML: Predicción multiclase de riesgo cardiovascular

Práctica 1 de la asignatura de Machine Learning (Máster en Inteligencia Artificial, UMU, curso 2026/27).

El objetivo es clasificar a pacientes adultos en tres niveles de riesgo cardiovascular (**0: bajo**, **1: moderado**, **2: alto**) a partir de una cohorte clínica de 2.254 pacientes con 27 variables sociodemográficas, clínicas, analíticas y de estilo de vida.

## Componentes del grupo

- Alberto Caballero Tello (alberto.caballerot@um.es)
- Cesar Martín Castellanos (cesar.m@um.es)
- Jose Francisco González (josefrancisco.gonzalem@um.es)

## Estructura del repositorio

| Fichero | Descripción |
|---|---|
| `ML-Entregable1.ipynb` | Notebook principal: EDA, preprocesado, modelos, evaluación y conclusiones |
| `enfermedad_cardiovascular.csv` | Dataset proporcionado |
| `requirements.txt` | Librerías y versiones exactas usadas |
| `data/splits/` | Particiones de desarrollo y test generadas por el notebook |
| `models/` | Modelos entrenados (por ahora, el baseline inicial, que el notebook ya no regenera) |
| `ML-MemoriaEntregable1.tex` / `.docx` | Memoria técnica (plantilla IEEE) |
| `Practica_1_Machine_Learning_Cardiovascular.pdf` | Enunciado de la práctica |
| `PR_CTICA_1-_Instrucciones.pdf` | Instrucciones de entrega |

## Cómo reproducirlo

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/alberct96/ML-Entregable1.git
   cd ML-Entregable1
   ```
2. Crear y activar un entorno virtual (Python 3.12 o superior):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux / Mac
   source .venv/bin/activate
   ```
3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Abrir `ML-Entregable1.ipynb` con Jupyter o VS Code y ejecutar todas las celdas en orden.

La semilla aleatoria está fijada (`SEED = 42`) para que los resultados sean reproducibles.

## Evolución del proyecto

### Semana 1 (26 oct - 1 nov): análisis exploratorio

- Preparado el entorno: `requirements.txt` con versiones fijadas y `.gitignore`.
- EDA completo en el apartado 1 del notebook. Problemáticas encontradas:
  - **Nulos** en `colesterol_ldl`, `glucosa_ayunas`, `horas_sueno_diarias` y `proteina_c_reactiva` (5-8% cada una, ~24% de las filas). Se reparten de forma parecida entre clases, así que se imputarán en vez de eliminar filas.
  - **Valores imposibles e incoherencias**: alcohol, HDL y triglicéridos negativos, IMC extremadamente bajo, menores de edad, más de 7 días de ejercicio por semana, sistólica ≤ diastólica, LDL > colesterol total...
  - **Valores repetidos anómalos**: sistólica, colesterol total, triglicéridos y ácido úrico tienen dos valores exactos repetidos decenas de veces en la cola alta (420 filas en total). Parecen outliers inyectados, no mediciones reales.
  - **Desbalance de clases** 50% / 30% / 20%, con la clase de alto riesgo como minoritaria.
  - **Variables categóricas** que hay que codificar (`nivel_actividad` es ordinal, las otras dos nominales).
  - Correlaciones bajas entre predictoras, así que no hay multicolinealidad.
- Decisión: todo el preprocesado que aprenda de los datos irá dentro de un `Pipeline` y se ajustará solo con train para evitar data leakage.

### Semana 2 (2 nov - 8 nov): revisión del EDA y primer pipeline

- Primera partición estratificada 68% / 17% / 15% (entrenamiento, validación y test), primer `Pipeline` de preprocesado y Random Forest como modelo baseline.
- Documentación completa del apartado 1 del notebook: cada análisis tiene su explicación y su observación, y las variables del código pasan a estar en inglés.
- Notebook preparado para ejecutarse de principio a fin desde un clon limpio. Añadido `joblib` a `requirements.txt`.
- **Rangos clínicos revisados**: los límites se justifican por definición de la variable, por la población de estudio o fisiológicamente, y solo marcan valores inverosímiles. Con ellos hay 263 valores fuera de rango en 11 de las 19 variables numéricas.
- Nuevos hallazgos y decisiones:
  - **Alcohol negativo** (177 filas, el 82,5% de clase 0): parece la cola baja de la variable, así que se recortará a 0 en lugar de imputarlo.
  - **Incoherencias** entre variables (por ejemplo, HDL + LDL > colesterol total en 559 filas o 123 fumadores actuales con 0 años fumando): no se corrigen por defecto, porque no hay un criterio fiable para saber qué valor es erróneo.
  - **Valores repetidos anómalos**: son el 91% de los outliers de sus variables (449 de 494). Requieren un tratamiento específico.
  - **Outliers de proteína C reactiva**: el 32,6% son de alto riesgo, frente al 20% global, así que no se aplicará una eliminación general de outliers.
  - **Variables casi independientes**: incluso las que clínicamente deberían ir juntas tienen correlaciones muy bajas, lo que sugiere un dataset sintético. Habrá que comprobar si los imputadores multivariantes mejoran a la mediana.
- Resumen de 10 problemáticas, con su evidencia y tratamiento propuesto, en el apartado 1.11.
- Reestructuración del preprocesado (apartado 2):
  - **Validación**: se descarta el conjunto de validación fijo, porque un único conjunto resulta poco fiable para comparar configuraciones. Se mantiene un test estratificado del 15% y la selección se hace con validación cruzada estratificada de 5 folds sobre el 85% restante (conjunto de desarrollo).
  - **Métricas de seguimiento**: macro-F1, exactitud balanceada y recall de la clase de alto riesgo.
  - **Valores fuera de rango**: alcohol y ejercicio se recortan siempre a su límite; en el resto de variables se implementan el recorte y la conversión a nulo para compararlos.
  - **Valores repetidos anómalos**: el recorte por percentiles no los corrige, porque el percentil 99 coincide con el máximo. Se localizan por su valor exacto, con detección ajustada solo en entrenamiento, y se convierten en nulos.

### Semana 3 (9 nov - 15 nov): imputación y desbalance

- **Nulos tras la limpieza**: cerca del 40% de los pacientes tiene algún valor ausente, así que se imputan en lugar de eliminar filas.
- **Métodos de imputación**: se comparan la mediana, la imputación iterativa y KNN con datos estandarizados, ocultando valores conocidos para medir el error de reconstrucción. La iterativa y la mediana obtienen un error prácticamente igual y KNN es peor, aunque los tres se mantienen para comparar su efecto sobre los modelos.
- **Indicador de valor ausente**: no se añade, porque con la corrección de Bonferroni ninguna variable muestra una relación significativa entre estar ausente y la clase.
- **Desbalance de clases**: se comparan el aprendizaje sensible al coste (`class_weight="balanced"`) y el remuestreo con `SMOTENC`, que respeta las variables categóricas. Ambos se aplican solo durante el entrenamiento.
- **Modelos (apartado 3)**: dos modelos base (regresión logística y SVM) y tres ensembles (Random Forest, XGBoost y LightGBM). Los hiperparámetros se ajustan con búsqueda aleatoria sobre la validación cruzada de 5 folds y la mejor configuración se elige por macro-F1. Las funciones `build_model_pipeline` y `tune_model` sirven para cualquier configuración de preprocesado, admiten `class_weight` y remuestreo, y guardan la puntuación de cada fold para los test estadísticos.
- **Primera comparación** con una configuración de referencia provisional: los cinco modelos quedan entre 0,648 y 0,664 de macro-F1. XGBoost es el mejor en macro-F1, pero los modelos base detectan más pacientes de alto riesgo (recall de 0,75 frente a 0,70).
- **Evaluación (apartado 4)**: justificación clínica de las métricas y una métrica adicional, el porcentaje de pacientes de alto riesgo clasificados como bajo riesgo (1,8% con la regresión logística frente a 4,4% con XGBoost). El análisis de errores se hace con predicciones out-of-fold, y el test se reserva para la configuración final.
- **Test estadísticos (apartado 5)**: test de Friedman y t-test pareado corregido de Nadeau y Bengio con ajuste de Holm. Con la configuración de referencia, ninguna diferencia entre modelos es significativa.
- Pendiente: montaje del pipeline con las configuraciones a comparar (apartado 2.6) y repetir con ellas el ajuste, la evaluación y los test; después, evaluación final en test y conclusiones.
