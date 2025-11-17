# app/main.py

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# ---------------------------------------------------------------------
# Configuración básica
# ---------------------------------------------------------------------

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Online News Popularity - Serving API",
    description=(
        "API de predicción para el proyecto de MLOps (Online News Popularity).\n\n"
        "Tarea 2: Serving y Portabilidad del Modelo con FastAPI.\n"
        "El modelo se carga automáticamente desde el artefacto más reciente "
        "en MCE/Model_construction_and_evaluation/mlruns/**/artifacts/model.pkl."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------
# Lista de features en el orden esperado por el schema de entrada
# (46 columnas definidas en Fase 1)
# ---------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "n_tokens_title",
    "n_tokens_content",
    "n_unique_tokens",
    "n_non_stop_words",
    "num_hrefs",
    "num_self_hrefs",
    "num_imgs",
    "num_videos",
    "num_keywords",
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
    "kw_min_min",
    "kw_max_min",
    "kw_min_max",
    "kw_max_max",
    "kw_min_avg",
    "kw_max_avg",
    "weekday_is_monday",
    "weekday_is_tuesday",
    "weekday_is_wednesday",
    "weekday_is_thursday",
    "weekday_is_friday",
    "weekday_is_saturday",
    "LDA_00",
    "LDA_01",
    "LDA_02",
    "LDA_03",
    "LDA_04",
    "global_subjectivity",
    "global_sentiment_polarity",
    "global_rate_positive_words",
    "global_rate_negative_words",
    "rate_positive_words",
    "rate_negative_words",
    "min_positive_polarity",
    "max_positive_polarity",
    "min_negative_polarity",
    "max_negative_polarity",
    "title_subjectivity",
    "title_sentiment_polarity",
    "abs_title_subjectivity",
    "abs_title_sentiment_polarity",
]

# ---------------------------------------------------------------------
# Esquemas Pydantic (entrada / salida)
# ---------------------------------------------------------------------


class OnlineNewsFeatures(BaseModel):
    # Texto / estructura del artículo
    n_tokens_title: float
    n_tokens_content: float
    n_unique_tokens: float
    n_non_stop_words: float

    # Links / multimedia
    num_hrefs: float
    num_self_hrefs: float
    num_imgs: float
    num_videos: float

    # Keywords
    num_keywords: float

    # Canales (one-hot)
    data_channel_is_lifestyle: int
    data_channel_is_entertainment: int
    data_channel_is_bus: int
    data_channel_is_socmed: int
    data_channel_is_tech: int
    data_channel_is_world: int

    # Estadísticos de keywords
    kw_min_min: float
    kw_max_min: float
    kw_min_max: float
    kw_max_max: float
    kw_min_avg: float
    kw_max_avg: float

    # Día de la semana (one-hot simplificado)
    weekday_is_monday: int
    weekday_is_tuesday: int
    weekday_is_wednesday: int
    weekday_is_thursday: int
    weekday_is_friday: int
    weekday_is_saturday: int

    # Tópicos LDA
    LDA_00: float
    LDA_01: float
    LDA_02: float
    LDA_03: float
    LDA_04: float

    # Polaridad / subjetividad
    global_subjectivity: float
    global_sentiment_polarity: float
    global_rate_positive_words: float
    global_rate_negative_words: float
    rate_positive_words: float
    rate_negative_words: float
    min_positive_polarity: float
    max_positive_polarity: float
    min_negative_polarity: float
    max_negative_polarity: float
    title_subjectivity: float
    title_sentiment_polarity: float
    abs_title_subjectivity: float
    abs_title_sentiment_polarity: float


class PredictionResponse(BaseModel):
    prediction: float
    model_path: str


# ---------------------------------------------------------------------
# Carga del modelo en startup
# ---------------------------------------------------------------------

MODEL: Optional[object] = None
LOADED_MODEL_PATH: Optional[Path] = None


def _find_latest_model_pkl(mlruns_dir: Path) -> Optional[Path]:
    """
    Busca recursivamente el 'model.pkl' más reciente dentro de mlruns/**/artifacts.
    """
    if not mlruns_dir.exists():
        logger.error(f"El directorio mlruns no existe: {mlruns_dir}")
        return None

    candidates = list(mlruns_dir.rglob("model.pkl"))
    if not candidates:
        logger.error(f"No se encontró ningún 'model.pkl' dentro de {mlruns_dir}")
        return None

    # Tomamos el más reciente por fecha de modificación
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = candidates[0]
    logger.info(f"Se encontró model.pkl más reciente en: {latest}")
    return latest


@app.on_event("startup")
def load_model_on_startup() -> None:
    """
    Evento de inicio de la aplicación: carga el modelo desde mlruns.
    """
    global MODEL, LOADED_MODEL_PATH

    # Repo root = .../MLops_team41
    repo_root = Path(__file__).resolve().parents[1]

    # Ruta al módulo MCE (según estructura del repo)
    mce_root = repo_root / "MCE" / "Model_construction_and_evaluation"
    mlruns_dir = mce_root / "mlruns"

    logger.info(f"Buscando modelo en: {mlruns_dir}")

    model_path = _find_latest_model_pkl(mlruns_dir)
    if model_path is None:
        # Detenemos el startup con una excepción clara
        raise RuntimeError(
            f"No se encontró ningún archivo 'model.pkl' en {mlruns_dir}. "
            "Asegúrate de haber corrido el pipeline de entrenamiento (dvc repro / script de MCE) "
            "para generar el artefacto del modelo."
        )

    try:
        MODEL = joblib.load(model_path)
        LOADED_MODEL_PATH = model_path
        logger.info(f"Modelo cargado exitosamente desde: {model_path}")
    except Exception as e:
        logger.exception("Error al cargar el modelo.")
        raise RuntimeError(
            f"No se pudo cargar el modelo desde {model_path}: {e}"
        ) from e


# ---------------------------------------------------------------------
# Helpers para construir el input al modelo
# ---------------------------------------------------------------------


def build_feature_vector(item: OnlineNewsFeatures) -> List[float]:
    """
    Construye el vector de 46 features para un registro,
    usando el orden definido en FEATURE_NAMES.
    """
    return [getattr(item, feat) for feat in FEATURE_NAMES]


def adjust_features_to_model(X: np.ndarray) -> np.ndarray:
    """
    Ajusta dinámicamente el número de columnas de X al que espera el modelo.
    - Si el modelo espera más columnas, se rellenan con ceros al final.
    - Si espera menos, se recortan columnas sobrantes.
    """
    global MODEL

    if MODEL is None:
        raise RuntimeError("El modelo aún no está cargado.")

    if hasattr(MODEL, "n_features_in_"):
        expected = int(MODEL.n_features_in_)
        current = X.shape[1]

        if current < expected:
            padding = np.zeros((X.shape[0], expected - current))
            X = np.concatenate([X, padding], axis=1)
        elif current > expected:
            X = X[:, :expected]

    return X


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------


@app.get("/")
def read_root():
    """
    Endpoint básico para verificar que el servicio está vivo.
    """
    return {
        "message": "API de Online News Popularity funcionando 🚀",
        "n_features_expected": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "model_path": str(LOADED_MODEL_PATH) if LOADED_MODEL_PATH else None,
    }


@app.post("/predict", response_model=List[PredictionResponse])
def predict(payload: List[OnlineNewsFeatures]):
    """
    Recibe una lista de registros (artículos) y regresa una predicción de 'shares' por cada uno.

    Ejemplo de body (uno solo):

    [
      {
        "n_tokens_title": 0.71,
        "n_tokens_content": -0.77,
        "n_unique_tokens": 0.89,
        "n_non_stop_words": 0.17,
        "num_hrefs": -0.71,
        "num_self_hrefs": -0.41,
        "num_imgs": -0.49,
        "num_videos": -0.41,
        "num_keywords": -0.14,
        "data_channel_is_lifestyle": 0,
        "data_channel_is_entertainment": 1,
        "data_channel_is_bus": 0,
        "data_channel_is_socmed": 0,
        "data_channel_is_tech": 0,
        "data_channel_is_world": 0,
        "kw_min_min": -0.43,
        "kw_max_min": -0.03,
        "kw_min_max": -0.05,
        "kw_max_max": -0.29,
        "kw_min_avg": -0.98,
        "kw_max_avg": -0.08,
        "weekday_is_monday": 1,
        "weekday_is_tuesday": 0,
        "weekday_is_wednesday": 0,
        "weekday_is_thursday": 0,
        "weekday_is_friday": 0,
        "weekday_is_saturday": 0,
        "LDA_00": 1.20,
        "LDA_01": 1.06,
        "LDA_02": -0.63,
        "LDA_03": -0.62,
        "LDA_04": -0.67,
        "global_subjectivity": 0.68,
        "global_sentiment_polarity": -0.00,
        "global_rate_positive_words": 0.23,
        "global_rate_negative_words": 0.00,
        "rate_positive_words": 0.46,
        "rate_negative_words": 0.00,
        "min_positive_polarity": 0.24,
        "max_positive_polarity": 0.06,
        "min_negative_polarity": -0.23,
        "max_negative_polarity": -0.71,
        "title_subjectivity": -0.27,
        "title_sentiment_polarity": 0.07,
        "abs_title_subjectivity": 0.69,
        "abs_title_sentiment_polarity": -0.97
      }
    ]
    """
    if MODEL is None or LOADED_MODEL_PATH is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo aún no está cargado. Revisa los logs de inicio "
                "para ver el problema al cargar el artefacto."
            ),
        )

    if not payload:
        raise HTTPException(
            status_code=400,
            detail="El cuerpo de la petición debe ser una lista no vacía de objetos de features.",
        )

    # 1) Construimos la matriz X con los 46 features
    try:
        rows = [build_feature_vector(item) for item in payload]
        X = np.array(rows, dtype=float)
    except AttributeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al construir el vector de features. Revisa los nombres de campos: {e}",
        )

    # 2) Ajustamos el número de columnas al que espera el modelo (ej. 60)
    try:
        X_adjusted = adjust_features_to_model(X)
    except Exception as e:
        logger.exception("Error al ajustar el tamaño de las features.")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al preparar el input del modelo: {e}",
        )

    # 3) Predicción
    try:
        y_pred = MODEL.predict(X_adjusted)
    except Exception as e:
        logger.exception("Error al hacer la predicción.")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al ejecutar el modelo: {e}",
        )

    return [
        PredictionResponse(
            prediction=float(pred),
            model_path=str(LOADED_MODEL_PATH),
        )
        for pred in y_pred
    ]