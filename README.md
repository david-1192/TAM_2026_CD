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
│   ├── PARCIAL_1_TAM_CD.ipynb        # Parcial 1 (Práctico): Validación numérica de Estimadores y MMD
│   └── PARCIAL_1_TAM_CD.pdf          # Parcial 1 (Teórico): Deducciones matemáticas en Regresión, RKHS y MMD
├── 03_TRABAJO-CLASES/                # Notebooks, laboratorios y apuntes de clase
│   ├── TAM_Taller_Alternativo.pdf    # Taller Alternativo (Teórico): Perceptrón, k-NN y Métricas de Clasificación
│   └── Taller_Alternativo.ipynb      # Taller Alternativo (Práctico): Implementación de Perceptrón, k-NN y Evaluación
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
- **Espacio de muestras ($N \times N$) y Kernels:** Evaluación de similitud entre observaciones mediante el **Kernel Lineal** y el **Kernel Gaussiano (RBF)** $K_{i,j} = \exp(-\gamma \|x_i - x_j\|^2)$), incluyendo la fundamentación matemática y visualización de mapas de calor para la covarianza ($Cov_K$) y correlación ($R_K$) en el espacio inducido por el kernel.

### 2. Taller 2: Regresión Polinomial y Paradigmas de Estimación (`TAM_TALLER2_CLASE.ipynb`)
Modelado, contaminación y reconstrucción de una señal armónica ($y(x) = \text{sen}(2\pi x)$) para contrastar enfoques frecuentistas y bayesianos:
- **Simulación de entorno ruidoso y expansión de base:** Adición de **Ruido Blanco Gaussiano Aditivo (AWGN)** controlado por $SNR_{dB}$ y construcción de la matriz de diseño polinomial de Vandermonde ($\Phi$).
- **Comparación de estimadores:** Análisis teórico y experimental de **Mínimos Cuadrados Ordinarios (OLS)**, **Regresión Ridge ($L_2$ / Tikhonov)**, **Máxima Verosimilitud (ML)** y **Máximo a Posteriori (MAP)**. Se demuestra el fenómeno de sobreajuste (*overfitting*) sin regularización y se verifica numéricamente la equivalencia $\text{OLS} \equiv \text{ML}$ (bajo ruido gaussiano) y $\text{Ridge} \equiv \text{MAP}$ (con prior gaussiano sobre los pesos).

---

## 📄 Evaluaciones Parciales (`02_PARCIALES/`)

### Parcial 1: Regresión en Espacios de Características, RKHS, MMD y Alineamiento de Kernels
Entrega compuesta por el documento teórico de demostraciones matemáticas (`PARCIAL_1_TAM_CD.pdf`) y el notebook de implementación y validación experimental (`PARCIAL_1_TAM_CD.ipynb`):

1. **Pregunta 1 — Regresión en Espacios de Características (`PDF` + `Notebook`):**
   - **Desarrollo Teórico:** Deducción matricial paso a paso de las soluciones cerradas para **OLS**, **Regresión Ridge ($L_2$)**, **Máxima Verosimilitud (MLE)** y **Máximo a Posteriori (MAP)**; análisis de condiciones de existencia, unicidad y convexidad estricta (matriz Hessiana $\mathbf{H} = \mathbf{\Phi}^\top\mathbf{\Phi}$), efecto estabilizador de $\lambda$ y demostración formal de las equivalencias $\text{OLS} \equiv \text{MLE}$ y $\text{Ridge} \equiv \text{MAP}$ ($\lambda = \sigma_\eta^2 / \sigma_w^2$).
   - **Validación Numérica:** Expansión polinomial no lineal de $P = 5$ a $Q = 20$ características ($N = 300$, partición 80/20), verificando computacionalmente la coincidencia exacta en $\text{MSE}_{\text{train}}$, $\text{MSE}_{\text{test}}$ y error de estimación $\|\hat{\mathbf{w}} - \mathbf{w}_*\|_2$ (diferencias del orden de precisión de máquina $\approx 10^{-16}$).

2. **Pregunta 2 — Espacios RKHS, Embeddings de Media y Discrepancia Máxima de Medias (MMD) (`PDF` + `Notebook`):**
   - **Desarrollo Teórico:** Definición rigurosa del *kernel mean embedding* $\mu_k = \mathbb{E}_{X}[\phi(X)] \in \mathcal{H}_k$ mediante la integral de Bochner, demostración de pertenencia al RKHS vía el **Teorema de Representación de Riesz**, e interpretación geométrica (inyectividad y codificación de momentos con kernels característicos). Deducción de la formulación poblacional, empírica sesgada ($\text{MMD}_b^2$) y su expresión cuadrática matricial $\frac{1}{n^2}\mathbf{1}_n^\top \mathbf{K}_{XX}\mathbf{1}_n + \frac{1}{m^2}\mathbf{1}_m^\top \mathbf{K}_{YY}\mathbf{1}_m - \frac{2}{nm}\mathbf{1}_n^\top \mathbf{K}_{XY}\mathbf{1}_m$.
   - **Experimento MNIST vs. Fashion-MNIST:** Evaluación empírica con muestras balanceadas ($n = m = 500$, $P = 784$) y kernel Gaussiano RBF sintonizado con la **Heurística de la Mediana** ($\sigma = 11.2255$), contrastando la discrepancia inter-distribución ($\approx 0.1316$) frente a comparaciones intra-distribución ($\approx 0.001$) y discutiendo el uso de pruebas de permutación.

3. **Pregunta 3 — Índice de Correlación de Pearson y Alineamiento Centrado en RKHS (`PDF`):**
   - **Desarrollo Teórico:** Formulación matricial del coeficiente de correlación de Pearson muestral $\hat{\rho}_{XY} = \frac{\mathbf{x}^\top\mathbf{H}\mathbf{y}}{\sqrt{(\mathbf{x}^\top\mathbf{H}\mathbf{x})(\mathbf{y}^\top\mathbf{H}\mathbf{y})}}$ empleando la matriz de centrado $\mathbf{H} = \mathbf{I}_N - \frac{1}{N}\mathbf{1}_N\mathbf{1}_N^\top$ (simétrica e idempotente).
   - **Alineamiento de Kernels:** Deducción del alineamiento centrado en RKHS a partir del producto interno de Frobenius y propiedades cíclicas de la traza: $\rho_{\text{RKHS}}(\mathbf{K}_X, \mathbf{K}_Y) = \frac{\text{tr}(\mathbf{K}_X\mathbf{H}\mathbf{K}_Y\mathbf{H})}{\sqrt{\text{tr}(\mathbf{K}_X\mathbf{H}\mathbf{K}_X\mathbf{H})\text{tr}(\mathbf{K}_Y\mathbf{H}\mathbf{K}_Y\mathbf{H})}}$, demostrando que para kernels lineales unidimensionales ($P=Q=1$) se cumple $\rho_{\text{RKHS}} = \hat{\rho}_{XY}^2$ y explicando la pérdida de orientación del signo.

---

## 💻 Trabajo en Clases (`03_TRABAJO-CLASES/`)

### Taller Alternativo: Clasificación Supervisada, Vecinos Cercanos y Evaluación de Modelos
Solución teórico-práctica (`TAM_Taller_Alternativo.pdf` y `Taller_Alternativo.ipynb`) centrada en algoritmos fundamentales de clasificación y métricas de desempeño:

1. **Punto 1 — Clasificación Lineal y Algoritmo del Perceptrón:**
   - Cálculo analítico y visualización de la frontera de decisión lineal ($w^\top x + b = 0$) para clasificación binaria.
   - Demostración matemática de la regla de actualización del perceptrón ($w_{\text{nuevo}} = w + \eta y_i x_i$, $b_{\text{nuevo}} = b + \eta y_i$), probando que el margen funcional de una muestra mal clasificada incrementa estrictamente en $m_{i,\text{nuevo}} = m_i + \eta(\|x_i\|_2^2 + 1) > m_i$, acompañado de verificación numérica paso a paso.
2. **Punto 2 — Clasificación por los $k$ Vecinos Más Cercanos ($k$-NN):**
   - Cálculo manual de distancias euclidianas y asignación por votación mayoritaria para distintos tamaños de vecindad ($k = 1, 3$).
   - Evaluación experimental sobre el conjunto sintético no lineal `make_moons` (partición estratificada 70/30) para $k \in \{1, 5, 15\}$, visualizando las regiones de decisión y analizando el compromiso sesgo-varianza (*overfitting* en $k=1$, suavizado y generalización óptima en $k=5$, y riesgo de *underfitting* al incrementar $k$).
3. **Punto 3 — Evaluación de un Clasificador Binario:**
   - Construcción manual y computacional (`scikit-learn`) de la **Matriz de Confusión** ($\text{TP}, \text{TN}, \text{FP}, \text{FN}$) y cálculo de **Accuracy** ($70\%$), **Precision** ($80\%$), **Recall / Sensibilidad** ($66.67\%$) y **$F_1$-score** ($72.73\%$).
   - Análisis crítico de escenarios donde la exactitud (*Accuracy*) resulta engañosa: conjuntos de datos altamente desbalanceados y problemas con costos de error asimétricos.

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
