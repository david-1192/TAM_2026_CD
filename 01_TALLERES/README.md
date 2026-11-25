# Talleres Prácticos (`01_TALLERES`)

Documentación detallada de los talleres prácticos desarrollados para la asignatura **Teoría de Aprendizaje de Máquina (2026)**.

[Volver al README principal](../README.md)

## Contenido del directorio

| Archivo | Tema principal |
| :--- | :--- |
| [TAM_TALLER1_CLASE.ipynb](./TAM_TALLER1_CLASE.ipynb) | Covarianza, correlación de Pearson y métodos de kernel (lineal y gaussiano RBF) sobre MNIST. |
| [TAM_TALLER2_CLASE.ipynb](./TAM_TALLER2_CLASE.ipynb) | Regresión polinomial con ruido AWGN y comparación entre estimadores OLS, Ridge, ML y MAP. |

---

## 1. Taller 1: Estadísticas Matriciales y Métodos de Kernel

Archivo asociado: [TAM_TALLER1_CLASE.ipynb](./TAM_TALLER1_CLASE.ipynb)

### Objetivo

Implementar desde cero formulaciones matriciales de estadísticas multivariadas (media, covarianza y correlación) en el espacio de características y extender el análisis al espacio de muestras mediante funciones de kernel lineal y gaussiano (RBF) sobre imágenes del conjunto de datos **MNIST**.

### Metodología y datos

- **Conjunto de datos:** MNIST ($28 \times 28$ píxeles aplanados a vectores de $P = 784$ características).
- **Espacio de características ($P \times P$):** Cálculo vectorizado sobre el conjunto completo de entrenamiento ($N = 60\,000$ muestras), generando matrices de dimensión $784 \times 784$.
- **Espacio de muestras ($N \times N$):** Para manejar los requerimientos de memoria RAM en Google Colab, se evalúan subconjuntos representativos:
  - **Kernel lineal:** subconjunto de $N = 2\,000$ muestras centradas, generando una matriz de similitud de $2\,000 \times 2\,000$.
  - **Kernel gaussiano (RBF):** subconjunto de $N = 3\,000$ muestras normalizadas en $[0, 1]$, con parámetro de escala $`\gamma = 1/P`$ ($`\gamma = 1/784`$), generando matrices de dimensión $3\,000 \times 3\,000$.

### Fundamentos matemáticos

Sea $`\mathbf{X} \in \mathbb{R}^{N \times P}`$ la matriz de datos de entrada con $N$ muestras y $P$ características.

#### Espacio de características

1. **Vector de medias y centrado de datos:**

```math
\boldsymbol{\mu} = \frac{1}{N}\mathbf{X}^{\top}\mathbf{1}_N, \qquad \mathbf{X}_c = \mathbf{X} - \mathbf{1}_N\boldsymbol{\mu}^{\top}
```

2. **Matriz de covarianza y correlación de Pearson ($784 \times 784$):**

```math
\mathrm{Cov} = \frac{1}{N}\mathbf{X}_c^{\top}\mathbf{X}_c, \qquad R_{i,j} = \frac{\mathrm{Cov}_{i,j}}{\sqrt{\mathrm{Cov}_{i,i}\,\mathrm{Cov}_{j,j}}}
```

En la implementación se añade un término pequeño de estabilidad numérica ($`\epsilon = 10^{-10}`$) para evitar divisiones por cero en píxeles de borde con varianza nula.

#### Espacio de muestras y métodos de kernel

1. **Kernel lineal:**

```math
\mathbf{K}_{\text{lin}} = \mathbf{X}_c \mathbf{X}_c^{\top}
```

2. **Matriz de Gram del kernel gaussiano (RBF):**

```math
K_{i,j} = \exp\!\left(-\gamma \lVert \mathbf{x}_i - \mathbf{x}_j \rVert_2^2\right)
```

3. **Centrado en el espacio inducido por el kernel (RKHS):** utilizando la matriz de centrado simétrica $`\mathbf{H} = \mathbf{I}_N - \frac{1}{N}\mathbf{1}_N\mathbf{1}_N^{\top}`$:

```math
\mathbf{K}_c = \mathbf{H}\mathbf{K}\mathbf{H} = \mathbf{K} - \frac{1}{N}\mathbf{1}_N\mathbf{1}_N^{\top}\mathbf{K} - \mathbf{K}\frac{1}{N}\mathbf{1}_N\mathbf{1}_N^{\top} + \frac{1}{N^2}\mathbf{1}_N\mathbf{1}_N^{\top}\mathbf{K}\mathbf{1}_N\mathbf{1}_N^{\top}
```

4. **Covarianza, varianzas y correlación en el espacio del kernel:**

```math
\mathrm{Cov}_K = \frac{1}{N}\mathbf{K}_c, \qquad \mathrm{Var}_K = \mathrm{diag}(\mathrm{Cov}_K) = \frac{1}{N}\mathrm{diag}(\mathbf{K}_c)
```

```math
\mathbf{R}_K = \mathbf{D}^{-1/2}\mathbf{K}_c\mathbf{D}^{-1/2}, \qquad (\mathbf{R}_K)_{i,j} = \frac{(\mathbf{K}_c)_{i,j}}{\sqrt{(\mathbf{K}_c)_{i,i} \cdot (\mathbf{K}_c)_{j,j}} + \epsilon}
```

donde $`\mathbf{D} = \mathrm{diag}(\mathrm{diag}(\mathbf{K}_c))`$ y $`\epsilon > 0`$ garantiza estabilidad numérica.

### Resultados obtenidos

- Se verificaron las dimensiones esperadas en cada etapa: vector de medias de $(784, 1)$, matrices de covarianza y correlación de características de $(784, 784)$, matriz de kernel lineal de $(2\,000, 2\,000)$ y matrices del kernel gaussiano RBF ($`\mathbf{K}`$, $`\mathbf{K}_c`$, $`\mathrm{Cov}_K`$ y $`\mathbf{R}_K`$) de $(3\,000, 3\,000)$.
- Se generaron mapas de calor comparativos para visualizar la estructura de correlación entre píxeles y la covarianza y correlación entre observaciones en el espacio inducido por el kernel gaussiano.

---

## 2. Taller 2: Regresión Polinomial y Paradigmas de Estimación

Archivo asociado: [TAM_TALLER2_CLASE.ipynb](./TAM_TALLER2_CLASE.ipynb)

### Objetivo

Modelar una señal armónica contaminada con ruido blanco gaussiano aditivo (AWGN) y reconstruirla mediante regresión polinomial, contrastando enfoques frecuentistas y bayesianos (**OLS**, **Ridge**, **ML** y **MAP**) para analizar el efecto de la regularización frente al sobreajuste.

### Metodología

1. **Señal de referencia:** muestreo uniforme de $N = 100$ puntos sobre el intervalo $x \in [0, 1]$ para la señal sinusoidal pura:

```math
y(x) = \mathrm{sen}(2\pi x)
```

2. **Simulación de ruido AWGN:** adición de ruido gaussiano $`n \sim \mathcal{N}(0, \sigma^2)`$ controlado por la relación señal-ruido deseada ($`\mathrm{SNR}_{\mathrm{dB}} = 3\text{ dB}`$):

```math
P_s = \frac{1}{N}\sum_{i=1}^{N} y_i^2, \qquad P_n = \frac{P_s}{10^{\mathrm{SNR}_{\mathrm{dB}}/10}}, \qquad t = y + n
```

3. **Expansión de base polinomial:** construcción de la matriz de diseño de Vandermonde $`\boldsymbol{\Phi}`$ para un polinomio de grado $10$ ($Q = 11$ términos de base):

```math
\boldsymbol{\Phi}(x) = \begin{bmatrix} 1 & x & x^2 & \dots & x^{Q-1} \end{bmatrix}
```

4. **Ajuste y comparación de estimadores:** evaluación de las soluciones analíticas sin regularización ($`\lambda = 0`$) y con regularización $L_2$ ($`\lambda = 0.1`$) sobre la señal ruidosa $t$.

### Fundamentos matemáticos

- **Mínimos Cuadrados Ordinarios (OLS) y Máxima Verosimilitud (ML):** minimizan directamente el error cuadrático empírico bajo ruido blanco gaussiano sin incorporar penalización sobre los parámetros ($`\lambda = 0`$):

```math
\hat{\mathbf{w}}_{\text{OLS}} = \hat{\mathbf{w}}_{\text{ML}} = (\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^{\top}\mathbf{t}
```

- **Regresión Ridge ($L_2$ / Tikhonov) y Máximo a Posteriori (MAP):** incorporan regularización cuadrática equivalente a asumir una distribución *a priori* gaussiana de media cero sobre los pesos ($`\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \sigma_w^2 \mathbf{I})`$):

```math
\hat{\mathbf{w}}_{\text{Ridge}} = \hat{\mathbf{w}}_{\text{MAP}} = (\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi} + \lambda \mathbf{I})^{-1}\boldsymbol{\Phi}^{\top}\mathbf{t}, \qquad \lambda = \frac{\sigma^2}{\sigma_w^2}
```

### Resultados y análisis

- **Sobreajuste en OLS / ML:** al emplear un polinomio de grado elevado sin regularización ($`\lambda = 0`$), el modelo se ajusta a las fluctuaciones del ruido en lugar de la tendencia armónica subyacente, produciendo pesos de magnitud elevada y oscilaciones pronunciadas en los extremos del dominio.
- **Estabilidad de Ridge / MAP:** la penalización cuadrática ($`\lambda = 0.1`$) restringe la norma de los coeficientes y estabiliza la inversión matricial, generando curvas más suaves que filtran las perturbaciones de alta frecuencia del ruido.
- **Equivalencia entre paradigmas:** se verifica de forma teórica y numérica la equivalencia $`\mathrm{OLS} \equiv \mathrm{ML}`$ bajo ruido gaussiano y $`\mathrm{Ridge} \equiv \mathrm{MAP}`$ bajo un *prior* gaussiano de media cero sobre los pesos.
