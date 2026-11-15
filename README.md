# 🧠 Teoría de Aprendizaje de Máquina (2026)

Repositorio académico correspondiente a la asignatura **Teoría de Aprendizaje de Máquina (Machine Learning Theory)** del periodo 2026. Aquí se centralizan talleres prácticos, evaluaciones parciales, ejercicios desarrollados en clase y el proyecto final del curso.

---

## 📂 Estructura del Repositorio

```text
TAM_2026/
├── 01_TALLERES/          # Talleres prácticos y entregas programadas
├── 02_PARCIALES/         # Evaluaciones parciales y soluciones
├── 03_TRABAJO-CLASES/    # Notebooks, laboratorios y apuntes de clase
├── 04_PROYECTO_FINAL/    # Desarrollo, documentación y código del proyecto final
├── .gitignore            # Archivos y directorios excluidos del control de versiones
├── README.md             # Documentación principal del repositorio
└── subir.sh              # Script automatizado para agregar, confirmar y subir cambios
```

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
