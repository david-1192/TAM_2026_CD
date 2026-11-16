# 💻 Trabajo en Clases (`03_TRABAJO-CLASES`)

Directorio destinado a los laboratorios prácticos, talleres en clase y experimentos comparativos desarrollados durante la asignatura **Teoría de Aprendizaje de Máquina (2026)**.

---

## 📂 Contenido del Directorio

| Archivo | Descripción |
| :--- | :--- |
| `RED_DENSA_CONVOLUCIONAL.ipynb` | Taller práctico comparativo entre Redes Neuronales Densas (MLP) y Redes Neuronales Convolucionales (CNN) sobre MNIST e inferencia en imágenes manuscritas externas. |
| `Taller_Alternativo.ipynb` | Desarrollo práctico del Taller Alternativo: Perceptrón, $k$-Vecinos Más Cercanos ($k$-NN) sobre `make_moons` y métricas de clasificación binaria. |
| `TAM_Taller_Alternativo.pdf` | Documento teórico con las demostraciones matemáticas y cálculos analíticos del Taller Alternativo. |

---

## 🧠 1. Taller Comparativo: Redes Densas vs. Redes Convolucionales (`RED_DENSA_CONVOLUCIONAL.ipynb`)

Este laboratorio implementa y contrasta dos paradigmas de aprendizaje profundo utilizando **TensorFlow / Keras** para el problema de clasificación multiclase de dígitos manuscritos (**MNIST**: $60{,}000$ muestras de entrenamiento y $10{,}000$ de prueba).

### 🔹 Preprocesamiento y Representación de Datos
- **Normalización:** Escalado de niveles de gris originales $[0, 255]$ al intervalo $[0, 1]$ en precisión `float32`.
- **Adecuación geométrica por modelo:**
  - **Modelo Denso (`x_train_dense`, `x_test_dense`):** Aplanamiento (*flattening*) de cada imagen a un vector unidimensional de $28 \times 28 = 784$ características.
  - **Modelo Convolucional (`x_train_conv`, `x_test_conv`):** Conservación de la topología bidimensional agregando el canal explícito de intensidad $(28, 28, 1)$.

### 🔹 Arquitecturas Implementadas
Ambos modelos fueron entrenados bajo las mismas condiciones experimentales: optimizador `Adam`, función de pérdida `sparse_categorical_crossentropy`, `epochs=15`, `batch_size=32` y partición de validación del $10\%$ (`validation_split=0.1`).

1. **Red Neuronal Densa (`Modelo_Denso`):**
   - `Input(shape=(784,))`
   - `Dense(64, activation='relu')`
   - `Dense(64, activation='relu')`
   - `Dense(10, activation='softmax')`

2. **Red Neuronal Convolucional (`Modelo_Convolucional`):**
   - `Input(shape=(28, 28, 1))`
   - `Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')` + `MaxPooling2D(pool_size=(2, 2))`
   - `Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')` + `MaxPooling2D(pool_size=(2, 2))`
   - `Flatten()`
   - `Dense(10, activation='softmax')`

### 📊 Resultados Experimentales

| Modelo | Loss (Entrenamiento) | Accuracy (Entrenamiento) | Loss (Prueba) | Accuracy (Prueba) |
| :--- | :---: | :---: | :---: | :---: |
| **Red Densa (`model_dense`)** | `0.013992` | `99.50%` | `0.103787` | `97.76%` |
| **Red Convolucional (`model_conv`)** | `0.004772` | `99.82%` | `0.047878` | **`98.92%`** |

- **Análisis:** La arquitectura convolucional reduce a menos de la mitad la pérdida en el conjunto de prueba (`0.0479` vs. `0.1038`) y disminuye el error de clasificación en más de un $50\%$ relativo ($1.08\%$ vs. $2.24\%$), gracias a la extracción jerárquica de patrones espaciales locales e invariancia a traslaciones inducida por las capas convolucionales y de *MaxPooling*.

### 🖼️ Inferencia con Imágenes Externas (`predecir_imagen_externa`)
Incluye un flujo completo de preprocesamiento e inferencia para evaluar dígitos manuscritos capturados fuera de MNIST:
1. Conversión a escala de grises (`L`) y redimensionamiento a $28 \times 28$ píxeles mediante interpolación `LANCZOS`.
2. Detección automática de fondo claro (media $> 127$) e inversión fotométrica (`ImageOps.invert`) para alinear la polaridad con MNIST (trazo claro sobre fondo negro).
3. Filtrado de ruido de fondo por umbralización suave (`< 0.2 -> 0.0`) y realce de contraste (`x1.3` con recorte en $[0, 1]$).
4. Visualización lado a lado de la imagen original vs. preprocesada y gráficas comparativas de la distribución de probabilidad (`Softmax`) predicha por ambas redes.

---

## 📐 2. Taller Alternativo: Clasificación Supervisada, $k$-NN y Evaluación (`Taller_Alternativo.ipynb` / `TAM_Taller_Alternativo.pdf`)

1. **Clasificación Lineal y Perceptrón:** Deducción analítica y verificación computacional del hiperplano separador y del incremento estricto del margen funcional tras una actualización del perceptrón.
2. **$k$-Vecinos Más Cercanos ($k$-NN):** Clasificación por distancias euclidianas y análisis del compromiso sesgo-varianza sobre el dataset no lineal `make_moons` con $k \in \{1, 5, 15\}$.
3. **Métricas de Desempeño en Clasificación Binaria:** Construcción de matriz de confusión, cálculo de *Accuracy*, *Precision*, *Recall* y *$F_1$-score*, y análisis de desbalance de clases y costos asimétricos.
