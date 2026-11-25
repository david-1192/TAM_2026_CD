# Teoría de Aprendizaje de Máquina (2026)

**Autor:** Cristian David Chalaca Salas

Repositorio académico correspondiente a la asignatura **Teoría de Aprendizaje de Máquina (2026)**. Centraliza los talleres prácticos, evaluaciones parciales, actividades desarrolladas en clase y el espacio destinado al proyecto final del curso, manteniendo la documentación técnica detallada dentro de cada directorio.

---

## Estructura del repositorio

```text
TAM_2026/
├── 01_TALLERES/          # Talleres prácticos y cuadernos de análisis
├── 02_PARCIALES/         # Evaluaciones parciales teóricas y prácticas
├── 03_TRABAJO-CLASES/    # Laboratorios, talleres en clase y comparativas
├── 04_PROYECTO_FINAL/    # Espacio reservado para el proyecto final
├── .gitignore            # Reglas de exclusión del control de versiones
├── README.md             # Presentación general e índice del repositorio
└── subir.sh              # Script de automatización para Git
```

---

## Contenido por carpeta

- [01_TALLERES/](./01_TALLERES/) — Talleres prácticos sobre estadísticas matriciales, métodos de kernel, regresión polinomial y regularización. Consultar [01_TALLERES/README.md](./01_TALLERES/README.md) para los detalles metodológicos y matemáticos.
- [02_PARCIALES/](./02_PARCIALES/) — Evaluaciones parciales que integran demostraciones teóricas en PDF y validaciones numéricas en notebooks sobre estimadores, RKHS, MMD y alineamiento de kernels. Consultar [02_PARCIALES/README.md](./02_PARCIALES/README.md).
- [03_TRABAJO-CLASES/](./03_TRABAJO-CLASES/) — Laboratorios y talleres en clase enfocados en clasificación supervisada (perceptrón, $k$-NN, métricas) y comparación entre redes neuronales densas y convolucionales. Consultar [03_TRABAJO-CLASES/README.md](./03_TRABAJO-CLASES/README.md).
- [04_PROYECTO_FINAL/](./04_PROYECTO_FINAL/) — Directorio destinado al futuro desarrollo y documentación del proyecto final del curso. Ver [04_PROYECTO_FINAL/README.md](./04_PROYECTO_FINAL/README.md).

---

## Herramientas y tecnologías

- **Lenguaje y notebooks:** Python 3.10+, Jupyter Notebook / JupyterLab.
- **Cálculo numérico y análisis de datos:** `numpy`, `scipy`, `pandas`.
- **Visualización:** `matplotlib`, `seaborn`.
- **Aprendizaje de máquina y profundo:** `scikit-learn`, `tensorflow` / `keras`.
- **Control de versiones:** Git y GitHub.

---

## Flujo de trabajo

1. **Google Colab:** desarrollo y ejecución de notebooks en entornos con aceleración de hardware para experimentos numéricos y entrenamiento de modelos.
2. **Ubuntu:** organización local del repositorio, revisión de archivos académicos, ejecución de scripts de soporte y gestión de versiones con Git.
3. **Antigravity CLI:** asistencia en el entorno local para estructuración, mantenimiento y verificación de la documentación del repositorio.
4. **GitHub:** alojamiento del repositorio remoto, trazabilidad del historial de entregas y visualización renderizada de notebooks y documentos Markdown.

---

## Uso del script `subir.sh`

El archivo [subir.sh](./subir.sh) automatiza la secuencia `git add .`, `git commit` y `git push origin main` para agilizar la publicación de actualizaciones en el repositorio remoto.

Con mensaje de commit personalizado:

```bash
./subir.sh "Descripción del cambio realizado"
```

Sin parámetros (asigna automáticamente la fecha y hora actual como mensaje de commit):

```bash
./subir.sh
```

Los archivos pesados de datos y modelos serializados (como `.csv`, `.zip`, `.pkl` y `.h5`) se mantienen excluidos del seguimiento mediante [.gitignore](./.gitignore).
