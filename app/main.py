import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import PatientInput
from app.database import (
    initialize_database,
    save_history,
    get_history_list,
    get_history_detail,
    delete_history
)


# ==========================================
# CAU HINH DUONG DAN
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "best_model.joblib"
CONTRACT_PATH = PROJECT_DIR / "feature_contract.json"
STATIC_DIR = PROJECT_DIR / "app" / "static"
INDEX_FILE = STATIC_DIR / "index.html"


# ==========================================
# NAP CUSTOM TRANSFORMER
# ==========================================

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import ckd_preprocessing


# ==========================================
# DOC FEATURE CONTRACT
# ==========================================

if not CONTRACT_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay Feature Contract: {CONTRACT_PATH}"
    )

with open(CONTRACT_PATH, "r", encoding="utf-8") as file:
    CONTRACT = json.load(file)

FEATURES = CONTRACT["input_features"]

if len(FEATURES) != 24 or CONTRACT["input_count"] != 24:
    raise RuntimeError("Feature Contract khong hop le.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# NAP DATABASE VA MO HINH
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    initialize_database()
    logger.info("Database lich su san sang.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay model: {MODEL_PATH}"
        )

    logger.info("Dang nap mo hinh CKD...")

    model = joblib.load(MODEL_PATH)

    classes = [
        int(value) for value in model.classes_
    ]

    if set(classes) != {0, 1}:
        raise RuntimeError(
            f"Tap nhan khong hop le: {classes}"
        )

    app.state.model = model

    logger.info(
        "Da nap mo hinh XGBoost Tuned thanh cong."
    )

    yield

    logger.info("Backend CKD da dung.")


# ==========================================
# KHOI TAO FASTAPI
# ==========================================

app = FastAPI(
    title="CKD Diagnosis API",
    description="API ho tro sang loc benh than man tinh",
    version="1.1.0",
    lifespan=lifespan
)


# ==========================================
# PHUC VU FILE GIAO DIEN
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# ==========================================
# TRANG CHU
# ==========================================

@app.get("/", include_in_schema=False)
async def home():

    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Khong tim thay file giao dien index.html."
        )

    return FileResponse(
        path=INDEX_FILE,
        media_type="text/html"
    )


# ==========================================
# KIEM TRA BACKEND
# ==========================================

@app.get("/health")
async def health(request: Request):

    model_loaded = hasattr(
        request.app.state,
        "model"
    )

    return {
        "status": "running",
        "model_loaded": model_loaded,
        "model_name": "XGBoost Tuned"
    }


# ==========================================
# API DU DOAN CKD
# ==========================================

@app.post("/predict")
async def predict(
    patient: PatientInput,
    request: Request
):

    if not hasattr(request.app.state, "model"):
        raise HTTPException(
            status_code=503,
            detail="Mo hinh AI chua san sang."
        )

    model = request.app.state.model

    try:

        # Lay du lieu dau vao
        patient_data = patient.model_dump()

        # Sap xep theo Feature Contract
        patient_df = pd.DataFrame(
            [patient_data],
            columns=FEATURES
        )

        # Du doan nhan
        pred_label = int(
            model.predict(patient_df)[0]
        )

        # Xac suat tung lop
        probabilities = model.predict_proba(
            patient_df
        )[0]

        class_order = [
            int(value) for value in model.classes_
        ]

        ckd_index = class_order.index(1)
        not_ckd_index = class_order.index(0)

        ckd_probability = float(
            probabilities[ckd_index]
        )

        not_ckd_probability = float(
            probabilities[not_ckd_index]
        )

        result_label = (
            "CKD"
            if pred_label == 1
            else "Not CKD"
        )

        # Luu lich su phan tich
        history_record = save_history(
            prediction=pred_label,
            prediction_label=result_label,
            ckd_probability=ckd_probability,
            not_ckd_probability=not_ckd_probability,
            model_name="XGBoost Tuned",
            input_data=patient_data
        )

        logger.info(
            "Da luu lich su phan tich ID=%s",
            history_record["id"]
        )

        return {
            "success": True,
            "prediction": pred_label,
            "prediction_label": result_label,
            "ckd_probability": ckd_probability,
            "not_ckd_probability": not_ckd_probability,
            "model_name": "XGBoost Tuned",
            "history_id": history_record["id"],
            "created_at": history_record["created_at"],
            "message": (
                "Mo hinh du doan CKD."
                if pred_label == 1
                else "Mo hinh du doan Not CKD."
            ),
            "disclaimer": (
                "Ket qua chi mang tinh chat ho tro sang loc, "
                "khong thay the chan doan cua bac si."
            )
        }

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception:

        logger.exception(
            "Loi trong qua trinh du doan CKD."
        )

        raise HTTPException(
            status_code=500,
            detail="Da xay ra loi trong qua trinh du doan."
        )


# ==========================================
# API LAY DANH SACH LICH SU
# ==========================================

@app.get("/history")
async def history_list(
    limit: int = Query(default=100, ge=1, le=200)
):

    try:
        records = get_history_list(limit)

        return {
            "success": True,
            "total": len(records),
            "items": records
        }

    except Exception:

        logger.exception(
            "Loi khi lay danh sach lich su."
        )

        raise HTTPException(
            status_code=500,
            detail="Khong the lay lich su phan tich."
        )


# ==========================================
# API XEM CHI TIET LICH SU
# ==========================================

@app.get("/history/{record_id}")
async def history_detail(record_id: int):

    record = get_history_detail(record_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Khong tim thay lan phan tich."
        )

    return {
        "success": True,
        "item": record
    }


# ==========================================
# API XOA LICH SU
# ==========================================

@app.delete("/history/{record_id}")
async def history_delete(record_id: int):

    deleted = delete_history(record_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Khong tim thay lan phan tich can xoa."
        )

    return {
        "success": True,
        "message": "Da xoa lich su phan tich."
    }