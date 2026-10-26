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
  - **Valores imposibles e incoherencias**: alcohol, HDL y triglicéridos negativos, IMC < 12, menores de edad, más de 7 días de ejercicio por semana, sistólica ≤ diastólica, LDL > colesterol total...
  - **Valores repetidos anómalos**: sistólica, colesterol total, triglicéridos y ácido úrico tienen dos valores exactos repetidos decenas de veces en la cola alta (420 filas en total). Parecen outliers inyectados, no mediciones reales.
  - **Desbalance de clases** 50% / 30% / 20%, con la clase de alto riesgo como minoritaria.
  - **Variables categóricas** que hay que codificar (`nivel_actividad` es ordinal, las otras dos nominales).
  - Correlaciones bajas entre predictoras, así que no hay multicolinealidad.
- Decisión: todo el preprocesado que aprenda de los datos irá dentro de un `Pipeline` y se ajustará solo con train para evitar data leakage.
