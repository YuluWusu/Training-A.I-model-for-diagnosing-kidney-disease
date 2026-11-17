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
# NAP CAC MODULE CUA UNG DUNG
# ==========================================

from app.schemas import PatientInput

from app.analytics import (
    build_dataset_analytics,
    build_tree_shap_calculation,
)

from app.sample_data import load_test_samples

from app.database import (
    initialize_database,
    save_history,
    get_history_list,
    get_history_detail,
    delete_history,
)


# ==========================================
# DOC FEATURE CONTRACT
# ==========================================

if not CONTRACT_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy Feature Contract: {CONTRACT_PATH}"
    )

with open(CONTRACT_PATH, "r", encoding="utf-8") as file:
    CONTRACT = json.load(file)

FEATURES = CONTRACT["input_features"]

if len(FEATURES) != 24 or CONTRACT["input_count"] != 24:
    raise RuntimeError(
        "Feature Contract không hợp lệ. "
        "Hệ thống cần đúng 24 thuộc tính đầu vào."
    )


# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ==========================================
# NAP DATABASE VA MO HINH
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Khoi tao SQLite
    initialize_database()

    logger.info("Database lich su san sang.")

    # Kiem tra model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình: {MODEL_PATH}"
        )

    logger.info("Dang nap mo hinh CKD...")

    model = joblib.load(MODEL_PATH)

    # Kiem tra tap nhan
    classes = [
        int(value)
        for value in model.classes_
    ]

    if set(classes) != {0, 1}:
        raise RuntimeError(
            f"Tập nhãn của mô hình không hợp lệ: {classes}"
        )

    app.state.model = model

    logger.info(
        "Da nap mo hinh XGBoost Tuned thanh cong."
    )

    # ======================================
    # TAI THONG KE DU LIEU
    # ======================================

    logger.info(
        "Dang tinh toan thong ke bo du lieu CKD..."
    )

    app.state.dataset_analytics = build_dataset_analytics(
        PROJECT_DIR,
        model
    )

    logger.info(
        "Da tinh toan thong ke bo du lieu CKD."
    )

    # ======================================
    # TAI 10 MAU TEST
    # ======================================

    logger.info(
        "Dang lay 5 mau CKD va 5 mau Not CKD tu tap Test..."
    )

    samples = load_test_samples(
        PROJECT_DIR,
        limit_per_class=5
    )

    # Kiem tra so luong
    ckd_count = sum(
        sample["actual_class"] == 1
        for sample in samples
    )

    not_ckd_count = sum(
        sample["actual_class"] == 0
        for sample in samples
    )

    if len(samples) != 10:
        raise RuntimeError(
            f"Can 10 mau du lieu nhung chi lay duoc {len(samples)} mau."
        )

    if ckd_count != 5 or not_ckd_count != 5:
        raise RuntimeError(
            "Phan bo mau khong dung: "
            f"CKD={ckd_count}, Not CKD={not_ckd_count}"
        )

    app.state.dataset_samples = samples

    logger.info(
        "Da nap thanh cong 10 mau Test: "
        "5 CKD va 5 Not CKD."
    )

    yield

    logger.info("Backend CKD da dung.")


# ==========================================
# KHOI TAO FASTAPI
# ==========================================

app = FastAPI(
    title="CKD Diagnosis API",
    description="API ho tro sang loc benh than man tinh",
    version="1.2.0",
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
            detail="Không tìm thấy file index.html."
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

    samples_loaded = hasattr(
        request.app.state,
        "dataset_samples"
    )

    return {
        "status": "running",
        "model_loaded": model_loaded,
        "samples_loaded": samples_loaded,
        "model_name": "XGBoost Tuned"
    }


# ==========================================
# API THONG KE BO DU LIEU
# ==========================================

@app.get("/dataset/summary")
async def dataset_summary(request: Request):

    summary = getattr(
        request.app.state,
        "dataset_analytics",
        None
    )

    if summary is None:
        raise HTTPException(
            status_code=503,
            detail="Thống kê bộ dữ liệu chưa sẵn sàng."
        )

    return {
        "success": True,
        "data": summary
    }


# ==========================================
# API LAY 10 MAU DU LIEU TEST
# ==========================================

@app.get("/dataset/samples")
async def dataset_samples(request: Request):

    samples = getattr(
        request.app.state,
        "dataset_samples",
        None
    )

    if samples is None:
        raise HTTPException(
            status_code=503,
            detail="Dữ liệu mẫu chưa sẵn sàng."
        )

    ckd_samples = [
        sample
        for sample in samples
        if sample["actual_class"] == 1
    ]

    not_ckd_samples = [
        sample
        for sample in samples
        if sample["actual_class"] == 0
    ]

    return {
        "success": True,

        "total": len(samples),

        "summary": {
            "total_ckd": len(ckd_samples),
            "total_not_ckd": len(not_ckd_samples),
        },

        "items": samples
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
            detail="Mô hình AI chưa sẵn sàng."
        )

    model = request.app.state.model

    try:

        # Lay du lieu dau vao
        patient_data = patient.model_dump()

        # Sap xep dung 24 thuoc tinh
        patient_df = pd.DataFrame(
            [patient_data],
            columns=FEATURES
        )

        # Du doan nhan
        pred_label = int(
            model.predict(patient_df)[0]
        )

        # Tinh xac suat
        probabilities = model.predict_proba(
            patient_df
        )[0]

        class_order = [
            int(value)
            for value in model.classes_
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

        # Giai thich bang TreeSHAP
        calculation = build_tree_shap_calculation(
            model,
            patient_df,
            CONTRACT
        )

        # Luu lich su
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

            "calculation": calculation,

            "model_name": "XGBoost Tuned",

            "history_id": history_record["id"],

            "created_at": history_record["created_at"],

            "message": (
                "Mô hình dự đoán nhóm có nhãn CKD."
                if pred_label == 1
                else "Mô hình dự đoán nhóm có nhãn Not CKD."
            ),

            "disclaimer": (
                "Kết quả chỉ mang tính chất hỗ trợ sàng lọc, "
                "không thay thế chẩn đoán của bác sĩ."
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
            detail="Đã xảy ra lỗi trong quá trình dự đoán."
        )


# ==========================================
# API LAY DANH SACH LICH SU
# ==========================================

@app.get("/history")
async def history_list(
    limit: int = Query(
        default=100,
        ge=1,
        le=200
    )
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
            detail="Không thể lấy lịch sử phân tích."
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
            detail="Không tìm thấy lượt phân tích."
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
            detail="Không tìm thấy lượt phân tích cần xóa."
        )

    return {
        "success": True,
        "message": "Đã xóa lịch sử phân tích."
    }