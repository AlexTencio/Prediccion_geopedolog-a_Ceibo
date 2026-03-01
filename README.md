#  Predicción Geopedológica — Cuenca del Río Ceibo

Clasificación de órdenes de suelo mediante ensamble de tres modelos de machine learning (Random Forest, XGBoost, CatBoost) con estrategias de regularización diversas y predicción espacial raster.

---

##  Requisitos del sistema

- Python 3.10+
- GDAL (se recomienda instalar via `conda-forge`)
- ~4 GB RAM libres para la predicción espacial (ajustable vía `batch_size` en el config)

---

##  Instalación

### 1. Clonar el repositorio
```bash
git https://github.com/AlexTencio/Prediccion_geopedolog-a_Ceibo.git
cd soil-prediction
```

### 2. Crear entorno conda (recomendado)
```bash
conda create -n geopedologia python=3.10
conda activate geopedologia
conda install -c conda-forge gdal rasterio geopandas
pip install -r requirements.txt
```

### 3. Alternativa: solo pip
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

##  Estructura del proyecto

```
soil-prediction/
├── soil_prediction.py       # Script principal
├── config.yaml              # ← Edita aquí tus rutas y parámetros
├── requirements.txt
├── .gitignore
├── README.md
├── data/                    # Tus archivos de datos (NO se suben a GitHub)
│   ├── General_calicatas_f.gpkg
│   ├── general.shp
│   └── rasters/
│       ├── NDVIGEN.tif
│       ├── GEOLOGIAGEN.tif
│       └── ...
└── outputs/                 # Generado automáticamente al ejecutar
```

> **Nota:** La carpeta `data/` y `outputs/` están en `.gitignore`. Los datos geoespaciales deben descargarse/copiarse manualmente (ver sección **Datos** más abajo).

---

##  Datos necesarios

| Archivo | Descripción |
|---|---|
| `General_calicatas_f.gpkg` | Puntos de calicatas con columna `Orden` (clase de suelo) |
| `general.shp` | Límite de la cuenca hidrográfica |
| `NDVIGEN.tif` | NDVI (Landsat/Sentinel) |
| `GEOLOGIAGEN.tif` | Mapa geológico rasterizado |
| `GEOMORFGEN.tif` | Mapa geomorfológico rasterizado |
| `CHIRTSGEN.tif` | Temperatura máxima (CHIRTS) |
| `CHIRPSGEN.tif` | Precipitación (CHIRPS) |
| `HILLSHADEGEN.tif` | Sombreado del relieve (DEM derivado) |
| `PENDIENTEGEN.tif` | Pendiente en grados (DEM derivado) |
| `DEM_GEN.tif` | Modelo Digital de Elevación |
| `RUGOSIDADGEN.tif` | Rugosidad del terreno (DEM derivado) |
| `GeomorphonsGEN.tif` | Geomorfones (GRASS r.geomorphon) |
| `USOGEN.tif` | Uso/cobertura del suelo |

---

##  Uso rápido

### 1. Configurar rutas

Edita `config.yaml` con las rutas absolutas o relativas a tus archivos:

```yaml
paths:
  output_dir: "outputs/"
  training_points: "data/General_calicatas_f.gpkg"
  watershed_shapefile: "data/general.shp"
  rasters:
    NDVI: "data/rasters/NDVIGEN.tif"
    # ... resto de variables
```

### 2. Ejecutar

```bash
python soil_prediction.py
# o con un config personalizado:
python soil_prediction.py --config mi_config.yaml
```

### 3. Revisar resultados

```
outputs/
├── puntos_train_test_split.gpkg         # Split espacial para verificación en QGIS
├── label_encoder_completo.joblib        # Mapeo clase ↔ entero
├── modelo_rf_diverso.joblib             # Modelo Random Forest entrenado
├── modelo_xgb_diverso.joblib            # Modelo XGBoost entrenado
├── modelo_cat_diverso.joblib            # Modelo CatBoost entrenado
├── cuenca_rf_30m_diverso.tif            # Mapa de suelos RF (30 m)
├── cuenca_xgb_30m_diverso.tif           # Mapa de suelos XGBoost (30 m)
├── cuenca_cat_30m_diverso.tif           # Mapa de suelos CatBoost (30 m)
├── confusion_matrix_*.png               # Matrices de confusión
├── feature_importance_*.png / *.csv     # Importancia de variables
└── comparacion_modelos_diverso.csv      # Métricas comparativas
```

---

##  Modelos y estrategias

| Modelo | Estrategia | Clave de diversidad |
|---|---|---|
| **Random Forest** | Conservadora | `max_samples=0.7`, alta regularización por hoja |
| **XGBoost** | Agresiva | `learning_rate=0.03`, muchos árboles pequeños |
| **CatBoost** | Balanceada | `l2_leaf_reg=10`, aleatorización fuerte en splits |

---

## 🛠️ Parámetros configurables

Todos los parámetros se controlan desde `config.yaml`, sin modificar el código:

```yaml
split:
  test_size: 0.30           # Proporción de test
  min_samples_per_class: 4  # Clases con menos muestras van al train

prediction:
  output_resolution_m: 30.0 # Resolución de salida en metros
  buffer_m: 500             # Buffer alrededor de la cuenca
  batch_size: 100000        # Reducir si hay problemas de memoria
  smoothing_kernel: 3       # Suavizado post-clasificación

models:
  random_forest:
    n_estimators: 150
    # ...
```

---

##  Verificación de resultados

Los rasters `.tif` pueden abrirse en **QGIS** o **ArcGIS**. Los valores de píxel corresponden a índices de clase (1-based); usa el `label_encoder_completo.joblib` para recuperar los nombres:

```python
import joblib
le = joblib.load("outputs/label_encoder_completo.joblib")
print(le.classes_)  # ['Entisol', 'Inceptisol', 'Mollisol', ...]
# Clase con índice 1 en el raster → le.classes_[0]
```

---

##  Cita / Referencia

Si usas este código en una publicación, por favor cita:

```
Remitirse a la tesis en formato digital o físico (bibliotecas de la Universidad de Costa Rica).
Tencio-Moya, A (2025). Modelado Geopedológico de la Cuenca del Río Ceibo
mediante Random Forest, XGBoost y CatBoost. Tesis: Clasificación supervisada de unidades geopedológicas para cuenca del río Ceibo, Buenos Aires, Puntarenas. https://github.com/AlexTencio/Prediccion_geopedolog-a_Ceibo.git
```

---

##  Licencia

MIT — libre para uso académico y comercial con atribución.
