# 🧠 Teoría de Aprendizaje de Máquina (2026)

Repositorio académico correspondiente a la asignatura **Teoría de Aprendizaje de Máquina (Machine Learning Theory)** del periodo 2026. Aquí se centralizan talleres prácticos, evaluaciones parciales, ejercicios desarrollados en clase y el proyecto final del curso.
## *CRISTIAN DAVID CHALACA SALAS*

---

## 📂 Estructura del Repositorio

```text
TAM_2026/
├── 01_TALLERES/                      # Talleres prácticos y entregas programadas
│   ├── TAM_TALLER1_CLASE.ipynb       # Taller 1: Covarianza, Correlación y Kernels (MNIST)
│   └── TAM_TALLER2_CLASE.ipynb       # Taller 2: Regresión Polinomial y Estimadores (OLS, Ridge, ML, MAP)
├── 02_PARCIALES/                     # Evaluaciones parciales y soluciones
├── 03_TRABAJO-CLASES/                # Notebooks, laboratorios y apuntes de clase
├── 04_PROYECTO_FINAL/                # Desarrollo, documentación y código del proyecto final
├── .gitignore                        # Archivos y directorios excluidos del control de versiones
├── README.md                         # Documentación principal del repositorio
└── subir.sh                          # Script automatizado para agregar, confirmar y subir cambios
```

---

## 📓 Descripción de Talleres (`01_TALLERES/`)

### 1. Taller 1: Estadísticas Matriciales y Métodos de Kernel (`TAM_TALLER1_CLASE.ipynb`)
Exploración de álgebra lineal estadística y funciones de kernel aplicadas al conjunto de datos **MNIST** ($28 \times 28 \rightarrow 784$ características):
- **Espacio de características ($P \times P$):** Formulación matricial desde cero del vector de medias ($\mu$), centrado de datos y cálculo de las matrices de covarianza y correlación de Pearson ($784 \times 784$).
- **Espacio de muestras ($N \times N$) y Kernels:** Evaluación de similitud entre observaciones mediante el **Kernel Lineal** y el **Kernel Gaussiano (RBF)** ($K_{i,j} = \exp(-\gamma \|x_i - x_j\|^2)$), incluyendo la fundamentación matemática y visualización de mapas de calor para la covarianza ($Cov_K$) y correlación ($R_K$) en el espacio inducido por el kernel.

### 2. Taller 2: Regresión Polinomial y Paradigmas de Estimación (`TAM_TALLER2_CLASE.ipynb`)
Modelado, contaminación y reconstrucción de una señal armónica ($y(x) = \text{sen}(2\pi x)$) para contrastar enfoques frecuentistas y bayesianos:
- **Simulación de entorno ruidoso y expansión de base:** Adición de **Ruido Blanco Gaussiano Aditivo (AWGN)** controlado por $SNR_{dB}$ y construcción de la matriz de diseño polinomial de Vandermonde ($\Phi$).
- **Comparación de estimadores:** Análisis teórico y experimental de **Mínimos Cuadrados Ordinarios (OLS)**, **Regresión Ridge ($L_2$ / Tikhonov)**, **Máxima Verosimilitud (ML)** y **Máximo a Posteriori (MAP)**. Se demuestra el fenómeno de sobreajuste (*overfitting*) sin regularización y se verifica numéricamente la equivalencia $\text{OLS} \equiv \text{ML}$ (bajo ruido gaussiano) y $\text{Ridge} \equiv \text{MAP}$ (con prior gaussiano sobre los pesos).

---

## 🛠️ Requisitos y Entorno de Trabajo

El código y los notebooks están desarrollados principalmente en **Python 3.x** utilizando herramientas estándar de Ciencia de Datos y Machine Learning:

- **Lenguaje:** Python 3.10+
- **Entorno interactivo:** Jupyter Notebook / JupyterLab
- **Librerías principales:**
  - `numpy`, `scipy` (Cálculo numérico y álgebra lineal)
  - `pandas` (Manipulación y análisis de datos)
  - `matplotlib`, `seaborn` (Visualización de datos)
  - `scikit-learn` (Modelos clásicos de Machine Learning)
  - `tensorflow` / `pytorch` (Aprendizaje profundo, según requerimiento)

### Instalación rápida de dependencias base

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn jupyterlab
```

---

## 🚀 Uso del Script de Automatización (`subir.sh`)

Para facilitar la actualización del repositorio en GitHub/GitLab, se incluye el script ejecutable `subir.sh`.

### Con mensaje personalizado:
```bash
./subir.sh "Entrega del Taller 1: Regresión Lineal y Regularización"
```

### Sin parámetros (usa automáticamente la fecha y hora actual):
```bash
./subir.sh
```

> **Nota:** Por buenas prácticas de almacenamiento, los conjuntos de datos pesados (`.csv`, `.zip`) y los pesos o modelos serializados (`.pkl`, `.h5`) están excluidos mediante el archivo `.gitignore`.
