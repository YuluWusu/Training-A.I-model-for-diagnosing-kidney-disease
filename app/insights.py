"""Giai thich bo sung cho ket qua du doan: chi so thieu, nguong tham chieu y khoa,
che do nguong quyet dinh (tau). Khong thay doi mo hinh, chi dien giai dau vao/dau ra.
"""
from __future__ import annotations

import math
from typing import Any

from app.analytics import FEATURE_LABELS, _get_fitted_parameters
from ckd_preprocessing import BINARY, FEATURES, MULTI, NUMERIC

STANDARD_TAU = 0.50
SCREENING_TAU = 0.25

CORE_FEATURES = ["hemo", "pcv", "sc", "al", "sg"]
HEMATOLOGY_CLUSTER = ["hemo", "pcv", "wc", "rc"]

# (thap, cao, don vi, mo ta khi thap, mo ta khi cao)
NUMERIC_RANGES = {
    "bp": (60, 80, "mmHg", "Huyết áp thấp", "Huyết áp tăng"),
    "bgr": (70, 140, "mg/dL", "Đường huyết thấp", "Đường huyết cao"),
    "bu": (15, 45, "mg/dL", "Urê máu thấp", "Urê máu cao"),
    "sc": (0.6, 1.3, "mg/dL", "Creatinine thấp", "Creatinine cao (giảm lọc cầu thận)"),
    "sod": (135, 145, "mEq/L", "Hạ natri máu", "Tăng natri máu"),
    "pot": (3.5, 5.0, "mEq/L", "Hạ kali máu", "Tăng kali máu"),
    "hemo": (12.0, 17.5, "g/dL", "Thiếu máu (Hemoglobin thấp)", "Hemoglobin cao"),
    "pcv": (36, 52, "%", "Dung tích hồng cầu thấp (thiếu máu)", "Dung tích hồng cầu cao"),
    "wc": (4000, 11000, "tế bào/mm³", "Bạch cầu thấp", "Bạch cầu cao (viêm/nhiễm trùng)"),
    "rc": (4.0, 6.0, "triệu/mm³", "Hồng cầu thấp (thiếu máu)", "Hồng cầu cao"),
}

ABNORMAL_CATEGORY = {
    "rbc": ("abnormal", "Hồng cầu niệu bất thường"),
    "pc": ("abnormal", "Tế bào mủ niệu bất thường"),
    "pcc": ("present", "Có cụm tế bào mủ"),
    "ba": ("present", "Có vi khuẩn niệu"),
    "htn": ("yes", "Có tăng huyết áp"),
    "dm": ("yes", "Có đái tháo đường"),
    "cad": ("yes", "Có bệnh động mạch vành"),
    "appet": ("poor", "Chán ăn / khẩu vị kém"),
    "pe": ("yes", "Có phù chân"),
    "ane": ("yes", "Có thiếu máu (ghi nhận lâm sàng)"),
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"", "?", "nan", "none", "null", "na", "n/a"}


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number == int(number) else f"{number:g}"


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _imputed_value(feature: str, params: dict[str, Any]) -> str:
    for item in params["numeric"]:
        if item["feature"] == feature:
            return _fmt(item["median"])
    for item in params["categorical"]:
        if item["feature"] == feature:
            return str(item["mode"])
    return "?"


def _reference_flag(feature: str, value: Any) -> dict[str, Any] | None:
    label = FEATURE_LABELS.get(feature, feature)

    if feature in NUMERIC_RANGES:
        number = _to_float(value)
        if number is None:
            return None
        low, high, unit, text_low, text_high = NUMERIC_RANGES[feature]
        ref = f"{_fmt(low)} – {_fmt(high)} {unit}"
        if number < low:
            return {"feature": feature, "label": label, "value": _fmt(number), "unit": unit,
                    "status": "low", "note": text_low, "reference": ref}
        if number > high:
            return {"feature": feature, "label": label, "value": _fmt(number), "unit": unit,
                    "status": "high", "note": text_high, "reference": ref}
        return None

    if feature == "sg":
        number = _to_float(value)
        if number is not None and number <= 1.010:
            return {"feature": feature, "label": label, "value": f"{number:.3f}", "unit": "",
                    "status": "low", "note": "Tỷ trọng thấp – giảm khả năng cô đặc nước tiểu của ống thận",
                    "reference": "1.015 – 1.025"}
        return None

    if feature in ("al", "su"):
        number = _to_float(value)
        if number is not None and number >= 1:
            note = ("Có đạm niệu – dấu ấn tổn thương màng lọc cầu thận"
                    if feature == "al" else "Có đường niệu – gợi ý đái tháo đường / tổn thương ống thận")
            return {"feature": feature, "label": label, "value": _fmt(number), "unit": "mức",
                    "status": "high", "note": note, "reference": "0"}
        return None

    if feature in ABNORMAL_CATEGORY:
        abnormal_value, note = ABNORMAL_CATEGORY[feature]
        if str(value).strip().lower() == abnormal_value:
            return {"feature": feature, "label": label, "value": str(value), "unit": "",
                    "status": "abnormal", "note": note, "reference": "Bình thường"}
    return None


def build_clinical_insight(model: Any, patient_data: dict[str, Any], ckd_probability: float) -> dict[str, Any]:
    params = _get_fitted_parameters(model)

    # 1. Chi so thieu va gia tri Pipeline tu dien khuyet
    missing = []
    for feature in FEATURES:
        if _is_missing(patient_data.get(feature)):
            missing.append({
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "imputed_value": _imputed_value(feature, params),
                "method": "median (Train)" if feature in NUMERIC else "mode (Train)",
                "is_core": feature in CORE_FEATURES,
            })

    missing_names = {item["feature"] for item in missing}
    core_missing = [item["feature"] for item in missing if item["is_core"]]
    cluster_missing = [name for name in HEMATOLOGY_CLUSTER if name in missing_names]

    warnings = []
    if len(cluster_missing) >= 3:
        warnings.append(
            "Thiếu đồng thời cụm xét nghiệm huyết học ("
            + ", ".join(cluster_missing)
            + "). Pipeline điền bằng trung vị của tập Train nên giá trị bị 'đồng hóa' về mức trung bình, "
              "độ bất định của dự đoán tăng cao."
        )
    if len(core_missing) >= 2:
        warnings.append(
            "Thiếu từ 2 chỉ số cốt lõi ("
            + ", ".join(core_missing)
            + ") – là các đặc trưng chi phối mạnh nhất của mô hình; kết quả chỉ mang tính tham khảo."
        )
    elif len(missing) >= 6:
        warnings.append(f"Hồ sơ thiếu {len(missing)}/24 chỉ số; hãy bổ sung xét nghiệm để kết quả đáng tin cậy hơn.")

    # 2. Doi chieu nguong tham chieu y khoa
    flags = []
    for feature in FEATURES:
        value = patient_data.get(feature)
        if _is_missing(value):
            continue
        flag = _reference_flag(feature, value)
        if flag:
            flags.append(flag)

    # 3. Che do nguong quyet dinh
    def verdict(tau: float) -> dict[str, Any]:
        positive = ckd_probability >= tau
        return {
            "tau": tau,
            "prediction": 1 if positive else 0,
            "prediction_label": "CKD" if positive else "Not CKD",
        }

    standard = verdict(STANDARD_TAU)
    screening = verdict(SCREENING_TAU)
    borderline = SCREENING_TAU <= ckd_probability < STANDARD_TAU

    return {
        "missing": {
            "count": len(missing),
            "total": len(FEATURES),
            "items": missing,
            "warnings": warnings,
            "uncertain": bool(warnings),
        },
        "reference_flags": flags,
        "threshold": {
            "standard": standard,
            "screening": screening,
            "borderline": borderline,
            "ckd_probability": ckd_probability,
            "note": (
                "Ca cận biên: ở ngưỡng chuẩn τ = 0,50 mô hình kết luận Not CKD, nhưng xác suất vẫn ≥ 25% – "
                "chế độ tầm soát (τ = 0,25) sẽ cảnh báo để bác sĩ xét nghiệm lại. Đây là cơ chế thu hồi các ca "
                "CKD giai đoạn sớm (như bệnh nhân ID = 102 ở Chương 4)."
                if borderline else
                "Kết luận ở hai ngưỡng τ = 0,50 và τ = 0,25 là "
                + ("giống nhau (không phải ca cận biên)." if standard["prediction"] == screening["prediction"]
                   else "khác nhau.")
            ),
        },
    }
