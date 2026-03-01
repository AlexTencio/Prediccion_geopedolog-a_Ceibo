"""
soil_prediction.py
==================
Modelado y Predicción Geopedológica - Cuenca del Río Ceibo
Versión: FINAL DIVERSO — compatible con GitHub

Uso:
    python soil_prediction.py                        # usa config.yaml por defecto
    python soil_prediction.py --config mi_config.yaml

Descripción:
    Entrena tres modelos de clasificación (Random Forest, XGBoost, CatBoost)
    con estrategias de regularización distintas para predecir órdenes de suelo
    sobre un área de cuenca a partir de variables ambientales raster.

Estructura de salidas:
    outputs/
    ├── puntos_train_test_split.gpkg
    ├── label_encoder_completo.joblib
    ├── modelo_rf_diverso.joblib
    ├── modelo_xgb_diverso.joblib
    ├── modelo_cat_diverso.joblib
    ├── cuenca_rf_30m_diverso.tif
    ├── cuenca_xgb_30m_diverso.tif
    ├── cuenca_cat_30m_diverso.tif
    ├── confusion_matrix_*.png
    ├── feature_importance_*.png / *.csv
    └── comparacion_modelos_diverso.csv
"""

import argparse
import os
import warnings

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import yaml
from catboost import CatBoostClassifier
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject
from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import xgboost as xgb

# Silenciar advertencias no críticas
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["CPL_LOG"] = "OFF"


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Carga el archivo YAML de configuración."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_raster_values(raster_info: list, coords: list, target_crs: str) -> np.ndarray:
    """
    Extrae valores de una lista de rasters en las coordenadas dadas.

    Parameters
    ----------
    raster_info : list of (name, path)
    coords      : list of (x, y) en target_crs
    target_crs  : CRS de las coordenadas de entrada

    Returns
    -------
    np.ndarray de forma (n_puntos, n_rasters)
    """
    columns = []
    for name, path in tqdm(raster_info, desc="Extrayendo valores de rasters"):
        with rasterio.open(path) as src:
            if src.crs.to_string() != rasterio.crs.CRS.from_string(target_crs).to_string():
                transformer = Transformer.from_crs(target_crs, src.crs, always_xy=True)
                coords_t = [transformer.transform(x, y) for x, y in coords]
                values = [val[0] for val in src.sample(coords_t)]
            else:
                values = [val[0] for val in src.sample(coords)]
        columns.append(values)
    return np.column_stack(columns)


def suavizar_clasificacion(pred_raster: np.ndarray, mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Aplica filtro de mediana para suavizar la clasificación espacial."""
    suavizado = median_filter(pred_raster, size=kernel_size)
    suavizado[~mask] = 0
    return suavizado.astype(np.uint8)


def guardar_matriz_confusion(
    y_true, y_pred, clases, model_name: str, output_dir: str, tipo: str = "test"
) -> str:
    """Genera y guarda la matriz de confusión como PNG."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clases, yticklabels=clases)
    plt.title(f"Matriz de Confusión - {model_name.upper()} ({tipo.upper()})")
    plt.ylabel("Clase Real")
    plt.xlabel("Clase Predicha")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"confusion_matrix_{model_name}_{tipo}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def guardar_importancia_variables(
    importancias: np.ndarray, nombres: list, model_name: str, output_dir: str
) -> pd.DataFrame:
    """Genera gráfico y CSV de importancia de variables."""
    df = (
        pd.DataFrame({"Variable": nombres, "Importancia": importancias})
        .sort_values("Importancia", ascending=False)
    )
    top_n = min(15, len(df))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), df["Importancia"].head(top_n), color=colors)
    plt.yticks(range(top_n), df["Variable"].head(top_n))
    plt.xlabel("Importancia")
    plt.title(f"Importancia de Variables - {model_name.upper()}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_importance_{model_name}.png"), dpi=300)
    plt.close()
    df.to_csv(os.path.join(output_dir, f"feature_importance_{model_name}.csv"), index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────

def build_models(cfg: dict) -> dict:
    """Instancia los tres clasificadores con los parámetros del config."""
    p = cfg["models"]
    return {
        "rf": RandomForestClassifier(
            **p["random_forest"],
            bootstrap=True,
            class_weight="balanced",
            n_jobs=-1,
            verbose=0,
        ),
        "xgb": xgb.XGBClassifier(
            **p["xgboost"],
            n_jobs=-1,
            verbosity=0,
        ),
        "cat": CatBoostClassifier(
            **p["catboost"],
            verbose=0,
            thread_count=-1,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FLUJO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    target_col = cfg["target_column"]
    projected_crs = cfg["prediction"].get("projected_crs", cfg.get("projected_crs", "EPSG:32617"))

    print("🗺️  MODELADO Y PREDICCIÓN GEOPEDOLÓGICA - CUENCA DEL RÍO CEIBO")
    print("=" * 70)
    print("📌 VERSIÓN FINAL — Modelos con estrategias diversas")
    print("=" * 70)

    # ── 1. Cargar puntos de entrenamiento ─────────────────────────────────────
    print("\n📍 1. Cargando puntos de entrenamiento...")
    puntos_path = cfg["paths"]["training_points"]
    gdf = gpd.read_file(puntos_path)
    print(f"   ✅ {len(gdf)} muestras cargadas")

    if target_col not in gdf.columns:
        raise ValueError(f"Columna '{target_col}' no encontrada. Columnas disponibles: {list(gdf.columns)}")

    n_orig = len(gdf)
    gdf = gdf[gdf[target_col].notna()].copy()
    if n_orig > len(gdf):
        print(f"   ⚠️  Eliminados {n_orig - len(gdf)} puntos con nulos en '{target_col}'")

    gdf["unique_id"] = range(len(gdf))

    distribucion = gdf[target_col].value_counts().sort_values(ascending=False)
    print(f"\n   📊 Distribución de clases ({len(distribucion)} clases):")
    for clase, count in distribucion.items():
        print(f"      {clase:25}: {count:4} pts ({count/len(gdf)*100:5.1f}%)")

    # ── 2. División Train / Test ───────────────────────────────────────────────
    print("\n🎲 2. Dividiendo datos en Train/Test...")
    min_s = cfg["split"]["min_samples_per_class"]
    clases_ok = distribucion[distribucion >= min_s].index
    clases_chicas = distribucion[distribucion < min_s]

    if len(clases_chicas) > 0:
        print(f"   ⚠️  Clases con < {min_s} muestras van íntegras al Train:")
        for c, n in clases_chicas.items():
            print(f"      {c}: {n} pts")
        gdf_ok = gdf[gdf[target_col].isin(clases_ok)]
        gdf_chicas = gdf[~gdf[target_col].isin(clases_ok)]
        gdf_train_ok, gdf_test = train_test_split(
            gdf_ok,
            test_size=cfg["split"]["test_size"],
            random_state=cfg["split"]["random_state"],
            stratify=gdf_ok[target_col],
        )
        gdf_train = pd.concat([gdf_train_ok, gdf_chicas], ignore_index=False)
    else:
        gdf_train, gdf_test = train_test_split(
            gdf,
            test_size=cfg["split"]["test_size"],
            random_state=cfg["split"]["random_state"],
            stratify=gdf[target_col],
        )

    print(f"   Train: {len(gdf_train)} pts  |  Test: {len(gdf_test)} pts")

    # Verificar data leakage
    ids_train = set(gdf_train["unique_id"])
    ids_test = set(gdf_test["unique_id"])
    inter = ids_train & ids_test
    if inter:
        raise ValueError(f"❌ DATA LEAKAGE: {len(inter)} muestras en train Y test.")
    print("   ✅ Sin intersección entre Train y Test")

    # Guardar split
    gdf_train["conjunto"] = "train"
    gdf_test["conjunto"] = "test"
    pd.concat([gdf_train, gdf_test]).to_file(
        os.path.join(out_dir, "puntos_train_test_split.gpkg"), driver="GPKG"
    )

    # ── 3. Cargar cuenca ──────────────────────────────────────────────────────
    print("\n📍 3. Cargando límites de la cuenca...")
    shp_path = cfg["paths"]["watershed_shapefile"]
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"No se encontró el shapefile: {shp_path}")
    gdf_cuenca = gpd.read_file(shp_path)
    # Reparar geometrías inválidas
    if not gdf_cuenca.geometry.is_valid.all():
        gdf_cuenca["geometry"] = gdf_cuenca.geometry.buffer(0)
    print("   ✅ Cuenca cargada")

    # ── 4. Variables raster ───────────────────────────────────────────────────
    print("\n🗺️  4. Configurando variables ambientales...")
    raster_dict = cfg["paths"]["rasters"]
    raster_info = list(raster_dict.items())   # [(name, path), ...]
    raster_names = [n for n, _ in raster_info]
    print(f"   ✅ {len(raster_names)} variables: {raster_names}")

    # ── 5. CRS de trabajo ─────────────────────────────────────────────────────
    print("\n🎯 5. Determinando CRS de trabajo...")
    target_crs = projected_crs if gdf_cuenca.crs.is_geographic else str(gdf_cuenca.crs)
    print(f"   ✅ CRS: {target_crs}")

    # ── 6. Extraer features ───────────────────────────────────────────────────
    print("\n📊 6. Extrayendo valores en puntos de entrenamiento...")
    for gdf_set in [gdf_train, gdf_test]:
        gdf_set.drop(
            gdf_set.index[~gdf_set.geometry.is_valid], inplace=True
        )

    gdf_train_w = gdf_train.to_crs(target_crs) if str(gdf_train.crs) != target_crs else gdf_train.copy()
    gdf_test_w = gdf_test.to_crs(target_crs) if str(gdf_test.crs) != target_crs else gdf_test.copy()

    coords_train = [(g.x, g.y) for g in gdf_train_w.geometry]
    coords_test = [(g.x, g.y) for g in gdf_test_w.geometry]

    X_train = extract_raster_values(raster_info, coords_train, target_crs)
    X_test = extract_raster_values(raster_info, coords_test, target_crs)
    y_train = gdf_train[target_col].values
    y_test = gdf_test[target_col].values

    print(f"   ✅ Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ── 7. Codificar etiquetas ────────────────────────────────────────────────
    print("\n🔤 7. Codificando clases de suelo...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    print(f"   Clases: {list(le.classes_)}")
    joblib.dump(le, os.path.join(out_dir, "label_encoder_completo.joblib"))

    # ── 8. Entrenar modelos ───────────────────────────────────────────────────
    print("\n🤖 8. Entrenando modelos con estrategias DIVERSAS...")
    modelos = build_models(cfg)
    for name, model in modelos.items():
        print(f"   Entrenando {name.upper()}...")
        model.fit(X_train, y_train_enc)
        joblib.dump(model, os.path.join(out_dir, f"modelo_{name}_diverso.joblib"))
        print(f"   ✅ {name.upper()} listo")

    # Verificar diversidad
    print("\n   🔍 Verificación de diversidad (20 primeras muestras de train):")
    preds_check = {n: m.predict(X_train[:20]) for n, m in modelos.items()}
    names_list = list(preds_check.keys())
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            a, b = names_list[i], names_list[j]
            diff = np.sum(preds_check[a] != preds_check[b])
            print(f"      {a.upper()} vs {b.upper()}: {diff}/20 diferencias ({diff/20*100:.0f}%)")

    # ── 9-10. Configurar predicción espacial ──────────────────────────────────
    print("\n🎯 9. Configurando área de predicción...")
    pred_cfg = cfg["prediction"]
    desired_res = pred_cfg["output_resolution_m"]
    buffer = pred_cfg["buffer_m"]

    gdf_cuenca_w = gdf_cuenca.to_crs(target_crs) if str(gdf_cuenca.crs) != target_crs else gdf_cuenca.copy()
    minx_b, miny_b, maxx_b, maxy_b = gdf_cuenca_w.total_bounds
    minx = minx_b - buffer
    miny = miny_b - buffer
    maxx = maxx_b + buffer
    maxy = maxy_b + buffer

    width = int((maxx - minx) / desired_res)
    height = int((maxy - miny) / desired_res)
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    print(f"   Dimensiones: {width}×{height} = {width*height:,} píxeles")

    mask_cuenca = geometry_mask(
        [g for g in gdf_cuenca_w.geometry if g is not None],
        out_shape=(height, width),
        transform=transform,
        invert=True,
    )
    print(f"   Píxeles en cuenca: {np.sum(mask_cuenca):,}")

    print("\n📊 10. Extrayendo datos de toda la cuenca...")
    raster_data = []
    for name, path in tqdm(raster_info, desc="Cargando rasters"):
        with rasterio.open(path) as src:
            if src.crs.to_string() != rasterio.crs.CRS.from_string(target_crs).to_string():
                data_out = np.zeros((height, width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=data_out,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    num_threads=4,
                )
            else:
                from rasterio.windows import from_bounds as win_from_bounds
                window = win_from_bounds(minx, miny, maxx, maxy, src.transform)
                data_out = src.read(1, window=window, out_shape=(height, width))
        data_out[~mask_cuenca] = 0
        raster_data.append(data_out)

    raster_stack = np.stack(raster_data, axis=-1)
    indices_cuenca = np.where(mask_cuenca)
    X_predict = raster_stack[indices_cuenca]
    print(f"   ✅ Píxeles a predecir: {len(X_predict):,}")
    del raster_data, raster_stack

    # ── 11. Validación y predicción ───────────────────────────────────────────
    print("\n🔍 11. Validando modelos y generando predicciones...")

    meta = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": width,
        "height": height,
        "count": 1,
        "crs": target_crs,
        "transform": transform,
        "compress": "lzw",
    }

    resultados = {}
    batch_size = pred_cfg["batch_size"]
    kernel_size = pred_cfg["smoothing_kernel"]

    for model_name, model in modelos.items():
        print(f"\n{'='*70}")
        print(f"MODELO: {model_name.upper()}")
        print(f"{'='*70}")

        # Train metrics
        y_pred_train = model.predict(X_train)
        acc_tr = accuracy_score(y_train_enc, y_pred_train)
        kappa_tr = cohen_kappa_score(y_train_enc, y_pred_train)
        print(f"   🎯 TRAIN  Acc={acc_tr:.4f}  Kappa={kappa_tr:.4f}")
        guardar_matriz_confusion(y_train_enc, y_pred_train, le.classes_, model_name, out_dir, "train")

        # Test metrics
        y_pred_test = model.predict(X_test)
        acc_te = accuracy_score(y_test_enc, y_pred_test)
        kappa_te = cohen_kappa_score(y_test_enc, y_pred_test)
        f1_macro = f1_score(y_test_enc, y_pred_test, average="macro")
        f1_weighted = f1_score(y_test_enc, y_pred_test, average="weighted")
        print(f"   🎯 TEST   Acc={acc_te:.4f}  Kappa={kappa_te:.4f}")
        print(f"   📊 Overfitting: {acc_tr - acc_te:.4f}")
        guardar_matriz_confusion(y_test_enc, y_pred_test, le.classes_, model_name, out_dir, "test")

        clases_en_test = np.unique(y_test_enc)
        print(
            classification_report(
                y_test_enc,
                y_pred_test,
                labels=clases_en_test,
                target_names=[le.classes_[i] for i in clases_en_test],
                digits=4,
                zero_division=0,
            )
        )

        if hasattr(model, "feature_importances_"):
            guardar_importancia_variables(model.feature_importances_, raster_names, model_name, out_dir)

        # Predicción espacial por batches
        print(f"   🗺️  Generando mapa espacial...")
        y_pred_cuenca = np.zeros(len(X_predict), dtype=np.int32)
        n_batches = int(np.ceil(len(X_predict) / batch_size))
        for i in tqdm(range(n_batches), desc="      Prediciendo"):
            s, e = i * batch_size, min((i + 1) * batch_size, len(X_predict))
            batch_pred = model.predict(X_predict[s:e])
            y_pred_cuenca[s:e] = batch_pred.flatten().astype(int)

        pred_raster = np.zeros((height, width), dtype=np.uint8)
        pred_raster[indices_cuenca] = y_pred_cuenca + 1
        pred_raster[~mask_cuenca] = 0

        pred_suavizado = suavizar_clasificacion(pred_raster, mask_cuenca, kernel_size)

        out_tif = os.path.join(out_dir, f"cuenca_{model_name}_30m_diverso.tif")
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(pred_suavizado, 1)
        print(f"   💾 Guardado: {os.path.basename(out_tif)}")

        resultados[model_name] = {
            "acc_train": acc_tr,
            "acc_test": acc_te,
            "kappa_test": kappa_te,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "overfitting": acc_tr - acc_te,
        }

    # ── 12. Resumen comparativo ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 RESUMEN COMPARATIVO")
    print("=" * 70)
    df_comp = pd.DataFrame(resultados).T
    print(df_comp.to_string())
    df_comp.to_csv(os.path.join(out_dir, "comparacion_modelos_diverso.csv"))

    if df_comp["acc_test"].nunique() == 1:
        print("\n⚠️  Todos los modelos tienen la misma accuracy.")
        print("   Esto es normal cuando los datos son perfectamente separables.")
    else:
        print("\n✅ Los modelos presentan métricas diferentes.")

    print(f"\n✅ PROCESO COMPLETADO")
    print(f"📁 Resultados en: {os.path.abspath(out_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicción Geopedológica - Cuenca del Río Ceibo")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Ruta al archivo de configuración YAML (default: config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
