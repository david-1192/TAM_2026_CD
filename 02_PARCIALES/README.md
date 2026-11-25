# Evaluaciones Parciales (`02_PARCIALES`)

Documentación detallada de las evaluaciones parciales desarrolladas en la asignatura **Teoría de Aprendizaje de Máquina (2026)**.

[Volver al README principal](../README.md)

## Contenido del directorio

| Archivo | Descripción |
| :--- | :--- |
| [PARCIAL_1_TAM_CD.pdf](./PARCIAL_1_TAM_CD.pdf) | Parcial 1 (Teórico): demostraciones matemáticas sobre regresión en espacios de características, RKHS, MMD, correlación de Pearson y alineamiento de kernels. |
| [PARCIAL_1_TAM_CD.ipynb](./PARCIAL_1_TAM_CD.ipynb) | Parcial 1 (Práctico): validación numérica de estimadores (OLS, Ridge, MLE y MAP) y experimento de MMD entre MNIST y Fashion-MNIST. |

---

## Parcial 1: Regresión en Espacios de Características, RKHS, MMD y Alineamiento de Kernels

Entrega compuesta por el documento teórico [PARCIAL_1_TAM_CD.pdf](./PARCIAL_1_TAM_CD.pdf) y el notebook de validación experimental [PARCIAL_1_TAM_CD.ipynb](./PARCIAL_1_TAM_CD.ipynb).

### Pregunta 1: Regresión en Espacios de Características

Archivos asociados: [PARCIAL_1_TAM_CD.pdf](./PARCIAL_1_TAM_CD.pdf) y [PARCIAL_1_TAM_CD.ipynb](./PARCIAL_1_TAM_CD.ipynb).

#### Desarrollo teórico

- Deducción matricial paso a paso de las soluciones en forma cerrada para **Mínimos Cuadrados Ordinarios (OLS)**, **Regresión Ridge ($L_2$)**, **Máxima Verosimilitud (MLE)** y **Máximo a Posteriori (MAP)**.
- Análisis de condiciones de existencia, unicidad y convexidad estricta mediante la matriz Hessiana $`\mathbf{H} = \boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}`$, así como el efecto estabilizador del parámetro de regularización $`\lambda > 0`$.
- Demostración formal de las equivalencias analíticas entre paradigmas frecuentistas y bayesianos:

```math
\hat{\mathbf{w}}_{\text{OLS}} = \hat{\mathbf{w}}_{\text{MLE}} = (\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^{\top}\mathbf{t}
```

```math
\hat{\mathbf{w}}_{\text{Ridge}} = \hat{\mathbf{w}}_{\text{MAP}} = (\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi} + \lambda \mathbf{I}_Q)^{-1}\boldsymbol{\Phi}^{\top}\mathbf{t}, \qquad \lambda = \frac{\sigma_{\eta}^2}{\sigma_w^2}
```

#### Validación numérica

Se realizó una expansión polinomial no lineal de $P = 5$ a $Q = 20$ características sobre $N = 300$ observaciones ($`\boldsymbol{\Phi} \in \mathbb{R}^{300 \times 20}`$), con partición 80/20 ($`N_{\text{train}} = 240`$, $`N_{\text{test}} = 60`$). Se fijó el nivel de ruido en $`\sigma_{\eta} = 0.5`$ ($`\sigma_{\eta}^2 = 0.25`$), el parámetro de regularización en $`\lambda = 1.0`$ y el *prior* gaussiano compatible $`\sigma_w^2 = \sigma_{\eta}^2 / \lambda = 0.25`$.

| Estimador | $`\mathrm{MSE}_{\text{train}}`$ | $`\mathrm{MSE}_{\text{test}}`$ | $`\lVert \hat{\mathbf{w}} - \mathbf{w}_* \rVert_2`$ |
| :--- | :---: | :---: | :---: |
| Mínimos Cuadrados (OLS) | `0.22085` | `0.26961` | `0.14877` |
| Máxima Verosimilitud (MLE) | `0.22085` | `0.26961` | `0.14877` |
| Mínimos Cuadrados Regularizados (Ridge) | `0.22133` | `0.27368` | `0.15352` |
| Máximo a Posteriori (MAP) | `0.22133` | `0.27368` | `0.15352` |

#### Análisis de resultados

- **Equivalencia OLS y MLE:** las métricas y el error de estimación coinciden exactamente ($`\lVert \hat{\mathbf{w}}_{\text{OLS}} - \hat{\mathbf{w}}_{\text{MLE}} \rVert_2 = 0.00 \times 10^0`$), confirmando que maximizar la log-verosimilitud bajo ruido gaussiano i.i.d. equivale a minimizar el error cuadrático ordinario.
- **Equivalencia Ridge y MAP:** al elegir $`\sigma_w^2 = \sigma_{\eta}^2 / \lambda`$, la matriz de penalización MAP se reduce analíticamente a $`\boldsymbol{\Phi}^{\top}\boldsymbol{\Phi} + \lambda \mathbf{I}_Q`$, obteniendo vectores de pesos idénticos ($`\lVert \hat{\mathbf{w}}_{\text{Ridge}} - \hat{\mathbf{w}}_{\text{MAP}} \rVert_2 = 0.00 \times 10^0`$).
- **Precisión numérica:** cualquier discrepancia del orden de $`10^{-16}`$ en aritmética de punto flotante corresponde únicamente a la precisión finita del estándar IEEE 754 en las operaciones matriciales de NumPy, y no a una diferencia teórica.

---

### Pregunta 2: Espacios RKHS, Kernel Mean Embedding y Maximum Mean Discrepancy (MMD)

Archivos asociados: [PARCIAL_1_TAM_CD.pdf](./PARCIAL_1_TAM_CD.pdf) y [PARCIAL_1_TAM_CD.ipynb](./PARCIAL_1_TAM_CD.ipynb).

#### Desarrollo teórico

- Definición del *kernel mean embedding* mediante la integral de Bochner:

```math
\mu_k = \mathbb{E}_{X}[\phi(X)] \in \mathcal{H}_k
```

- Demostración de la pertenencia al espacio de Hilbert con núcleo reproductor (RKHS) empleando el **Teorema de Representación de Riesz**, e interpretación geométrica de su inyectividad y codificación de momentos cuando el kernel es característico.
- Deducción de la formulación poblacional de la Discrepancia Máxima de Medias (MMD), su estimador empírico sesgado ($`\widehat{\mathrm{MMD}}_b^2`$) y su forma cuadrática matricial:

```math
\widehat{\mathrm{MMD}}_b^2 = \frac{1}{n^2}\mathbf{1}_n^{\top}\mathbf{K}_{XX}\mathbf{1}_n + \frac{1}{m^2}\mathbf{1}_m^{\top}\mathbf{K}_{YY}\mathbf{1}_m - \frac{2}{nm}\mathbf{1}_n^{\top}\mathbf{K}_{XY}\mathbf{1}_m
```

#### Validación experimental: MNIST vs. Fashion-MNIST

Se evaluaron muestras independientes sin etiquetas de $n = m = 500$ imágenes normalizadas en $[0, 1]$ ($P = 784$ características), generando matrices de Gram $`\mathbf{K}_{XX}, \mathbf{K}_{YY}, \mathbf{K}_{XY} \in \mathbb{R}^{500 \times 500}`$. Se utilizó el kernel gaussiano RBF:

```math
k(\mathbf{u}, \mathbf{v}) = \exp\!\left(-\frac{\lVert \mathbf{u} - \mathbf{v} \rVert_2^2}{2\sigma^2}\right)
```

con ancho de banda sintonizado mediante la **Heurística de la Mediana** sobre la muestra combinada, obteniendo $`\sigma = 11.2255`$.

| Comparación de distribuciones | $`\widehat{\mathrm{MMD}}_b^2`$ |
| :--- | :---: |
| MNIST (G1) vs. Fashion-MNIST (G1) | `0.131625` |
| MNIST (G1) vs. MNIST (G2 - disjunto) | `0.001640` |
| Fashion-MNIST (G1) vs. Fashion-MNIST (G2 - disjunto) | `0.001077` |

#### Conclusiones metodológicas

- **Separabilidad de distribuciones:** la discrepancia inter-distribución entre MNIST y Fashion-MNIST ($`\approx 0.1316`$) es casi dos órdenes de magnitud superior a las comparaciones intra-distribución ($`\approx 0.001`$), evidenciando la capacidad discriminativa del *kernel mean embedding* en el RKHS.
- **Sesgo muestral:** las comparaciones entre subconjuntos disjuntos de una misma distribución no son exactamente cero debido al tamaño muestral finito ($n = 500$) y al sesgo positivo de orden $`\mathcal{O}(1/n)`$ asociado a los términos diagonales $`k(\mathbf{x}_i, \mathbf{x}_i) = 1`$.
- **Pruebas de hipótesis:** un valor puntual de MMD requiere complementarse con pruebas de permutación (*permutation tests*) para aproximar la distribución nula empírica bajo $`H_0: P_X = P_Y`$ y obtener un *p*-valor estadísticamente confiable.

---

### Pregunta 3: Correlación de Pearson y Alineamiento Centrado de Kernels en RKHS

Archivo asociado: [PARCIAL_1_TAM_CD.pdf](./PARCIAL_1_TAM_CD.pdf).

#### Desarrollo teórico

1. **Correlación de Pearson en forma matricial:** utilizando la matriz de centrado $`\mathbf{H} = \mathbf{I}_N - \frac{1}{N}\mathbf{1}_N\mathbf{1}_N^{\top}`$ (simétrica e idempotente):

```math
\hat{\rho}_{XY} = \frac{\mathbf{x}^{\top}\mathbf{H}\mathbf{y}}{\sqrt{(\mathbf{x}^{\top}\mathbf{H}\mathbf{x})(\mathbf{y}^{\top}\mathbf{H}\mathbf{y})}}
```

2. **Alineamiento centrado de kernels en RKHS:** deducido a partir del producto interno de Frobenius y las propiedades cíclicas de la traza:

```math
\rho_{\mathrm{RKHS}}(\mathbf{K}_X, \mathbf{K}_Y) = \frac{\operatorname{tr}(\mathbf{K}_X \mathbf{H} \mathbf{K}_Y \mathbf{H})}{\sqrt{\operatorname{tr}(\mathbf{K}_X \mathbf{H} \mathbf{K}_X \mathbf{H})\,\operatorname{tr}(\mathbf{K}_Y \mathbf{H} \mathbf{K}_Y \mathbf{H})}}
```

3. **Relación en el caso lineal unidimensional ($P = Q = 1$):** se demuestra que para kernels lineales unidimensionales se cumple:

```math
\rho_{\mathrm{RKHS}}(\mathbf{K}_X, \mathbf{K}_Y) = \hat{\rho}_{XY}^2
```

lo cual explica la pérdida de información sobre el signo (orientación) de la correlación al pasar al alineamiento centrado de matrices de Gram.
