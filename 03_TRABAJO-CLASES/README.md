# Trabajo en Clases (`03_TRABAJO-CLASES`)

Documentación detallada de los laboratorios prácticos, talleres en clase y experimentos comparativos desarrollados durante la asignatura **Teoría de Aprendizaje de Máquina (2026)**.

[Volver al README principal](../README.md)

## Contenido del directorio

| Archivo | Descripción |
| :--- | :--- |
| [TAM_Taller_Alternativo.pdf](./TAM_Taller_Alternativo.pdf) | Documento teórico con demostraciones matemáticas y cálculos analíticos del Taller Alternativo (perceptrón, $k$-NN y métricas de clasificación). |
| [Taller_Alternativo.ipynb](./Taller_Alternativo.ipynb) | Implementación práctica del Taller Alternativo: frontera de decisión del perceptrón, $k$-NN sobre `make_moons` y evaluación de clasificación binaria. |
| [RED_DENSA_CONVOLUCIONAL.ipynb](./RED_DENSA_CONVOLUCIONAL.ipynb) | Taller comparativo entre una Red Neuronal Densa (MLP) y una Red Neuronal Convolucional (CNN) sobre MNIST, incluyendo inferencia con imágenes externas. |

---

## 1. Taller Alternativo: Clasificación Supervisada, Vecinos Cercanos y Métricas de Evaluación

Archivos asociados: [TAM_Taller_Alternativo.pdf](./TAM_Taller_Alternativo.pdf) y [Taller_Alternativo.ipynb](./Taller_Alternativo.ipynb).

### Punto 1: Clasificación lineal y algoritmo del perceptrón

- Cálculo analítico y visualización de la frontera de decisión lineal para clasificación binaria:

```math
\mathbf{w}^{\top}\mathbf{x} + b = 0
```

- Demostración matemática y verificación numérica paso a paso de la regla de actualización del perceptrón ante una muestra mal clasificada ($`(\mathbf{x}_i, y_i)`$):

```math
\mathbf{w}_{\text{nuevo}} = \mathbf{w} + \eta y_i \mathbf{x}_i, \qquad b_{\text{nuevo}} = b + \eta y_i
```

- Prueba de que el margen funcional de la muestra mal clasificada incrementa estrictamente tras la actualización:

```math
m_{i,\text{nuevo}} = m_i + \eta\left(\lVert \mathbf{x}_i \rVert_2^2 + 1\right) > m_i
```

### Punto 2: Clasificación por los k vecinos más cercanos (k-NN)

- Cálculo manual de distancias euclidianas y clasificación por votación mayoritaria para distintos tamaños de vecindad ($k = 1$ y $k = 3$).
- Evaluación experimental sobre el conjunto sintético no lineal `make_moons` ($300$ muestras, `noise=0.20`, partición estratificada 70/30) para $`k \in \{1, 5, 15\}`$, visualizando las regiones de decisión y analizando el compromiso sesgo-varianza:

| Vecinos ($k$) | Accuracy (Entrenamiento) | Accuracy (Prueba) | Comportamiento observado |
| :---: | :---: | :---: | :--- |
| `1` | `1.0000` | `0.9333` | Sobreajuste (*overfitting*) y frontera altamente irregular. |
| `5` | `0.9571` | `0.9556` | Suavizado adecuado y mejor desempeño de generalización. |
| `15` | `0.9571` | `0.9333` | Mayor suavizado con tendencia al subajuste (*underfitting*) al incrementar $k$. |

### Punto 3: Evaluación de un clasificador binario

- Construcción manual y verificación en `scikit-learn` de la **Matriz de Confusión** ($`\mathrm{TN} = 3`$, $`\mathrm{FP} = 1`$, $`\mathrm{FN} = 2`$, $`\mathrm{TP} = 4`$) sobre un conjunto binario de 10 observaciones:

| Métrica | Valor |
| :--- | :---: |
| Accuracy | `70.00 %` (`0.7000`) |
| Precision | `80.00 %` (`0.8000`) |
| Recall (Sensibilidad) | `66.67 %` (`0.6667`) |
| $F_1$-score | `72.73 %` (`0.7273`) |

- **Análisis crítico:** se examinan escenarios en los que la exactitud (*Accuracy*) resulta insuficiente o engañosa, en particular en conjuntos de datos altamente desbalanceados y en problemas con costos de error asimétricos entre falsos positivos y falsos negativos.

---

## 2. Taller Comparativo: Redes Neuronales Densas (MLP) vs. Redes Convolucionales (CNN)

Archivo asociado: [RED_DENSA_CONVOLUCIONAL.ipynb](./RED_DENSA_CONVOLUCIONAL.ipynb).

Este laboratorio implementa, entrena y contrasta dos arquitecturas de aprendizaje profundo en **TensorFlow / Keras** para la clasificación multiclase de dígitos manuscritos del conjunto **MNIST** ($60\,000$ imágenes de entrenamiento y $10\,000$ de prueba).

### Preprocesamiento y representación de datos

- **Normalización:** escalado de intensidades de píxeles desde el rango original $[0, 255]$ al intervalo $[0, 1]$ en precisión `float32`.
- **Adecuación dimensional según arquitectura:**
  - **Red Densa (`x_train_dense`, `x_test_dense`):** aplanamiento (*flattening*) de cada imagen a un vector unidimensional de $28 \times 28 = 784$ características en $`\mathbb{R}^{784}`$.
  - **Red Convolucional (`x_train_conv`, `x_test_conv`):** conservación de la estructura espacial bidimensional agregando el canal explícito de intensidad $(28, 28, 1)$.

### Arquitecturas e hiperparámetros de entrenamiento

Ambos modelos fueron entrenados bajo las mismas condiciones experimentales:

- **Optimizador:** `Adam`
- **Función de pérdida:** `sparse_categorical_crossentropy`
- **Épocas:** `15`
- **Tamaño de lote (`batch_size`):** `32`
- **Partición de validación (`validation_split`):** `0.1` (10 %)

1. **Red Neuronal Densa (`model_dense` / `Modelo_Denso`):**
   - `Input(shape=(784,))`
   - `Dense(64, activation='relu')`
   - `Dense(64, activation='relu')`
   - `Dense(10, activation='softmax')`

2. **Red Neuronal Convolucional (`model_conv` / `Modelo_Convolucional`):**
   - `Input(shape=(28, 28, 1))`
   - `Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')` + `MaxPooling2D(pool_size=(2, 2))`
   - `Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')` + `MaxPooling2D(pool_size=(2, 2))`
   - `Flatten()`
   - `Dense(10, activation='softmax')`

### Resultados experimentales

| Modelo | Loss (Entrenamiento) | Accuracy (Entrenamiento) | Loss (Prueba) | Accuracy (Prueba) |
| :--- | :---: | :---: | :---: | :---: |
| Red Densa (`model_dense`) | `0.013992` | `99.50 %` | `0.103787` | `97.76 %` |
| Red Convolucional (`model_conv`) | `0.004772` | `99.82 %` | `0.047878` | `98.92 %` |

- **Análisis comparativo:** la red convolucional reduce a menos de la mitad la pérdida en prueba (`0.0479` frente a `0.1038`) y disminuye el error de clasificación en más de un 50 % relativo (1.08 % frente a 2.24 %). Esta mejora evidencia la ventaja de preservar la estructura espacial local y extraer características jerárquicas mediante filtros convolucionales y capas de *MaxPooling*.

### Inferencia con imágenes externas

El notebook implementa la función automatizada `predecir_imagen_externa` para evaluar dígitos manuscritos capturados fuera de MNIST mediante el siguiente flujo:

1. Conversión a escala de grises (`L`) y redimensionamiento a $28 \times 28$ píxeles con interpolación `LANCZOS`.
2. Detección automática de fondo claro (media $> 127$) e inversión fotométrica (`ImageOps.invert`) para igualar la polaridad de MNIST (trazo claro sobre fondo oscuro).
3. Filtrado de ruido de fondo por umbralización suave (`< 0.2 -> 0.0`) y realce de contraste (`x1.3` con recorte en $[0, 1]$).
4. Visualización comparativa de la imagen original frente a la preprocesada junto con la distribución de probabilidades (`Softmax`) predicha por ambas arquitecturas.
