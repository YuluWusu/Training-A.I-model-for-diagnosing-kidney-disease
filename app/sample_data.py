from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ckd_preprocessing import FEATURES, NUMERIC


# Số mẫu cần lấy cho mỗi lớp
SAMPLES_PER_CLASS = 5


def normalize_class(value: Any) -> int:
    """
    Chuẩn hóa nhãn thực tế thành:
        1 = CKD
        0 = Not CKD

    Hỗ trợ cả nhãn dạng số và dạng chữ.
    """

    if pd.isna(value):
        raise ValueError("Phát hiện nhãn bị thiếu trong tập Test.")

    text = str(value).strip().lower()

    if text in {"1", "1.0", "ckd"}:
        return 1

    if text in {"0", "0.0", "notckd", "not ckd", "not_ckd"}:
        return 0

    raise ValueError(
        f"Không nhận diện được nhãn phân lớp: {value}"
    )


def normalize_feature_value(
    feature: str,
    value: Any
) -> float | str | None:
    """
    Chuyển dữ liệu CSV về kiểu phù hợp với PatientInput.

    - Giá trị thiếu được giữ là None.
    - Thuộc tính số được chuyển thành float.
    - Thuộc tính phân loại được chuyển thành chuỗi.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.lower() in {"", "?", "nan", "none", "null"}:
        return None

    if feature in NUMERIC:
        try:
            number = float(text)

        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Giá trị thuộc tính {feature} không hợp lệ: {value}"
            ) from error

        if not np.isfinite(number):
            return None

        return number

    # Các thuộc tính dạng phân loại như:
    # sg, al, su, rbc, pc, pcc, ba, htn, dm, cad,
    # appet, pe, ane.
    if isinstance(value, (float, np.floating)):
        return format(float(value), "g")

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    return text.lower()


def select_evenly_spaced_positions(
    positions: np.ndarray,
    number_of_samples: int
) -> list[int]:
    """
    Chọn mẫu trải đều theo vị trí trong từng lớp,
    bảo đảm kết quả ổn định giữa các lần chạy.
    """

    if len(positions) < number_of_samples:
        raise ValueError(
            f"Tập Test chỉ có {len(positions)} mẫu, "
            f"không đủ để lấy {number_of_samples} mẫu."
        )

    if number_of_samples == 1:
        return [int(positions[0])]

    selected_indexes = np.linspace(
        0,
        len(positions) - 1,
        number_of_samples,
        dtype=int
    )

    return [
        int(positions[index])
        for index in selected_indexes
    ]


def load_test_samples(
    project_dir: Path,
    limit_per_class: int = SAMPLES_PER_CLASS
) -> list[dict[str, Any]]:
    """
    Lấy dữ liệu mẫu từ tập Test độc lập.

    Nguồn:
        data/processed/X_test_raw.csv
        data/processed/y_test.csv

    Kết quả:
        5 mẫu CKD
        5 mẫu Not CKD

    Không tự tạo hoặc thay đổi giá trị thuộc tính.
    """

    processed_dir = project_dir / "data" / "processed"

    x_path = processed_dir / "X_test_raw.csv"
    y_path = processed_dir / "y_test.csv"

    # Kiểm tra file dữ liệu
    if not x_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu X Test: {x_path}"
        )

    if not y_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy nhãn y Test: {y_path}"
        )

    # Đọc dữ liệu đầu vào và nhãn thực tế
    X_test = pd.read_csv(x_path)
    y_frame = pd.read_csv(y_path)

    # Kiểm tra đủ 24 thuộc tính
    missing_features = [
        feature
        for feature in FEATURES
        if feature not in X_test.columns
    ]

    if missing_features:
        raise ValueError(
            "X_test_raw.csv thiếu các thuộc tính: "
            + ", ".join(missing_features)
        )

    # Lấy cột nhãn
    if "classification" in y_frame.columns:
        y_series = y_frame["classification"]

    elif y_frame.shape[1] == 1:
        y_series = y_frame.iloc[:, 0]

    else:
        raise ValueError(
            "Không xác định được cột classification trong y_test.csv."
        )

    # Kiểm tra số lượng dòng
    if len(X_test) != len(y_series):
        raise ValueError(
            "Số dòng X_test và y_test không bằng nhau: "
            f"{len(X_test)} và {len(y_series)}."
        )

    # Chuẩn hóa nhãn thành 0 và 1
    labels = np.array([
        normalize_class(value)
        for value in y_series
    ])

    # Lấy vị trí các mẫu thuộc từng lớp
    ckd_positions = np.flatnonzero(labels == 1)
    not_ckd_positions = np.flatnonzero(labels == 0)

    # Chọn đúng 5 mẫu mỗi lớp
    selected_ckd = select_evenly_spaced_positions(
        ckd_positions,
        limit_per_class
    )

    selected_not_ckd = select_evenly_spaced_positions(
        not_ckd_positions,
        limit_per_class
    )

    samples: list[dict[str, Any]] = []

    # Xử lý lần lượt hai nhóm
    groups = [
        {
            "class_value": 1,
            "class_name": "Có bệnh (CKD)",
            "prefix": "CKD",
            "positions": selected_ckd,
        },
        {
            "class_value": 0,
            "class_name": "Không bệnh (Not CKD)",
            "prefix": "NOT-CKD",
            "positions": selected_not_ckd,
        },
    ]

    for group in groups:

        for sample_number, row_position in enumerate(
            group["positions"],
            start=1
        ):
            row = X_test.iloc[row_position]

            # Lấy đủ 24 thuộc tính theo đúng thứ tự Feature Contract
            patient_features = {
                feature: normalize_feature_value(
                    feature,
                    row[feature]
                )
                for feature in FEATURES
            }

            # Đếm số trường thiếu
            missing_fields = sum(
                value is None
                for value in patient_features.values()
            )

            samples.append({
                "sample_code": (
                    f"{group['prefix']}-{sample_number:02d}"
                ),

                "actual_class": group["class_value"],

                "actual_label": group["class_name"],

                "split": "Test độc lập",

                "test_position": row_position,

                "missing_fields": missing_fields,

                "features": patient_features,
            })

    # Kiểm tra số lượng cuối cùng
    expected_total = limit_per_class * 2

    if len(samples) != expected_total:
        raise RuntimeError(
            f"Cần {expected_total} mẫu nhưng chỉ tạo được {len(samples)}."
        )

    actual_ckd = sum(
        item["actual_class"] == 1
        for item in samples
    )

    actual_not_ckd = sum(
        item["actual_class"] == 0
        for item in samples
    )

    if (
        actual_ckd != limit_per_class
        or actual_not_ckd != limit_per_class
    ):
        raise RuntimeError(
            "Số lượng mẫu của hai nhóm không đúng yêu cầu."
        )

    return samples



def _parse_uci857_midpoint(
    value: Any,
    low_default: float | None = None,
    high_default: float | None = None,
) -> float | None:
    """Chuyển chuỗi khoảng giá trị của UCI 857 sang điểm đại diện để suy luận."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if "-" in text:
        parts = [float(x.strip()) for x in text.split("-", 1)]
        return round((parts[0] + parts[1]) / 2.0, 2)
    match = re.search(r"([0-9.]+)", text)
    number = float(match.group(1)) if match else None
    if text.startswith("<") and low_default is not None:
        return low_default
    if text.startswith("≥") and high_default is not None:
        return high_default
    return number


def _uci857_row_to_features(row: pd.Series) -> dict[str, Any]:
    """Ánh xạ một bệnh án thật từ Enam Medical College (UCI 857) sang 24 thuộc tính chuẩn."""
    sg_val = _parse_uci857_midpoint(row["sg"], 1.005, 1.025)
    sg_levels = [1.005, 1.01, 1.015, 1.02, 1.025]
    sg_best = min(sg_levels, key=lambda lv: abs(lv - (sg_val or 1.02)))

    al_val = int(round(_parse_uci857_midpoint(row["al"], 0.0, 4.0) or 0.0))
    su_val = int(round(_parse_uci857_midpoint(row["su"], 0.0, 4.0) or 0.0))
    sc_text = str(row["sc"]).strip()
    sc_val = None if sc_text == "< 3.65" else _parse_uci857_midpoint(sc_text, 1.0, 20.0)

    raw_features = {
        "age": _parse_uci857_midpoint(row["age"], 10.0, 80.0),
        "bp": None,  # UCI 857 mã hóa huyết áp dạng 0/1 thay vì mmHg
        "sg": format(sg_best, "g"),
        "al": str(max(0, min(5, al_val))),
        "su": str(max(0, min(5, su_val))),
        "rbc": "abnormal" if str(row["rbc"]).strip() == "1" else "normal",
        "pc": "abnormal" if str(row["pc"]).strip() == "1" else "normal",
        "pcc": "present" if str(row["pcc"]).strip() == "1" else "notpresent",
        "ba": "present" if str(row["ba"]).strip() == "1" else "notpresent",
        "bgr": _parse_uci857_midpoint(row["bgr"], 95.0, 450.0),
        "bu": _parse_uci857_midpoint(row["bu"], 30.0, 240.0),
        "sc": sc_val,
        "sod": _parse_uci857_midpoint(row["sod"], 115.0, 155.0),
        "pot": _parse_uci857_midpoint(row["pot"], 4.2, 43.0),
        "hemo": _parse_uci857_midpoint(row["hemo"], 6.5, 16.8),
        "pcv": _parse_uci857_midpoint(row["pcv"], 20.0, 50.0),
        "wc": _parse_uci857_midpoint(row["wbcc"], 4500.0, 20000.0),
        "rc": _parse_uci857_midpoint(row["rbcc"], 2.5, 6.5),
        "htn": "yes" if str(row["htn"]).strip() == "1" else "no",
        "dm": "yes" if str(row["dm"]).strip() == "1" else "no",
        "cad": "yes" if str(row["cad"]).strip() == "1" else "no",
        "appet": "poor" if str(row["appet"]).strip() == "1" else "good",
        "pe": "yes" if str(row["pe"]).strip() == "1" else "no",
        "ane": "yes" if str(row["ane"]).strip() == "1" else "no",
    }
    return {feat: raw_features[feat] for feat in FEATURES}


def load_external_and_clinical_samples(project_dir: Path) -> list[dict[str, Any]]:
    """
    Nạp thêm các bộ mẫu kiểm định ngoại vi (UCI 857 - Enam Medical College, Bangladesh)
    và các hồ sơ lâm sàng / ca cận biên phục vụ đối chứng thực tế trên Web.
    """
    samples: list[dict[str, Any]] = []

    # 1. Mẫu bệnh nhân thực tế từ Bệnh viện Enam Medical College, Bangladesh (UCI ID 857)
    uci857_path = project_dir / "data" / "external" / "uci_857_ckd_dataset_v2.csv"
    if uci857_path.exists():
        ext_df = pd.read_csv(uci857_path).iloc[2:].reset_index(drop=True)
        selected_ext = [
            (0, "EXT-BD-S1", "CKD giai đoạn S1 (eGFR ≥ 227,9; al = 1; hemo = 11,3–12,6)"),
            (41, "EXT-BD-S3", "CKD giai đoạn S3 (eGFR 51,8–76,9; al ≥ 4; htn = yes; hemo = 10–11,3)"),
            (17, "EXT-BD-S4", "CKD giai đoạn S4 (eGFR < 26,6; sc = 3,65–6,8; hemo = 11,3–12,6)"),
            (23, "EXT-BD-S5", "CKD giai đoạn S5 cuối (sc = 16,25–19,4; hemo < 6,1; htn & dm = yes)"),
            (5, "EXT-BD-OK1", "Người khỏe mạnh (Not CKD, S1; hemo ≥ 16,5; al = 0; sg ≥ 1,023)"),
            (10, "EXT-BD-OK2", "Người khỏe mạnh (Not CKD, S2; hemo = 12,6–13,9; al = 0; sg ≥ 1,023)"),
        ]
        for row_idx, code, desc in selected_ext:
            if row_idx >= len(ext_df):
                continue
            row = ext_df.iloc[row_idx]
            feats = _uci857_row_to_features(row)
            cls_val = normalize_class(row["class"])
            samples.append({
                "sample_code": code,
                "group": "external_uci857",
                "source_name": "Ngoại vi — BV Enam (Bangladesh, UCI #857)",
                "actual_class": cls_val,
                "actual_label": "Có bệnh (CKD)" if cls_val == 1 else "Không bệnh (Not CKD)",
                "clinical_note": desc + " [quy đổi trung điểm khoảng UCI 857; bp=trống do nhị phân gốc]",
                "missing_fields": sum(v is None for v in feats.values()),
                "features": feats,
            })

    # 2. Ca cận biên thực tế & Hồ sơ lâm sàng chuẩn KDIGO
    x_test_path = project_dir / "data" / "processed" / "X_test_raw.csv"
    if x_test_path.exists():
        x_test = pd.read_csv(x_test_path)
        for pos, code, cls_val, desc in [
            (
                52,
                "DEMO-BORDER-102",
                1,
                "Ca cận biên ID=102 (17 tuổi, CKD sớm chưa thiếu máu: hemo=13,9, al=0, sc=2,1). Demo gạt ngưỡng τ=0,50 → 0,25!",
            ),
            (
                53,
                "DEMO-MISSING-324",
                0,
                "Ca thiếu cụm huyết học ID=324 (bỏ trống hemo, pcv, wc, rc). Demo cảnh báo 'Đồng hóa về trung vị'!",
            ),
        ]:
            if pos < len(x_test):
                row = x_test.iloc[pos]
                feats = {f: normalize_feature_value(f, row[f]) for f in FEATURES}
                samples.append({
                    "sample_code": code,
                    "group": "clinical_stress",
                    "source_name": "Tập Test thực tế — Ca phân tích lỗi Chương 4",
                    "actual_class": cls_val,
                    "actual_label": "Có bệnh (CKD)" if cls_val == 1 else "Không bệnh (Not CKD)",
                    "clinical_note": desc,
                    "missing_fields": sum(v is None for v in feats.values()),
                    "features": feats,
                })

    clinical_vignettes = [
        {
            "sample_code": "KDIGO-G3B-A3",
            "group": "clinical_stress",
            "source_name": "Hồ sơ lâm sàng chuẩn KDIGO (Ngoài Dataset)",
            "actual_class": 1,
            "actual_label": "Có bệnh (CKD)",
            "clinical_note": "Bệnh thận đái tháo đường giai đoạn G3b-A3 (61 tuổi, THA + ĐTĐ, sc=2,4, al=3, sg=1,010, hemo=10,2).",
            "features": {
                "age": 61.0, "bp": 95.0, "sg": "1.01", "al": "3", "su": "2",
                "rbc": "normal", "pc": "abnormal", "pcc": "notpresent", "ba": "notpresent",
                "bgr": 215.0, "bu": 64.0, "sc": 2.4, "sod": 134.0, "pot": 5.1,
                "hemo": 10.2, "pcv": 31.0, "wc": 8900.0, "rc": 3.6,
                "htn": "yes", "dm": "yes", "cad": "no", "appet": "poor", "pe": "yes", "ane": "yes",
            },
        },
        {
            "sample_code": "CLIN-GYM-OK",
            "group": "clinical_stress",
            "source_name": "Hồ sơ lâm sàng chuẩn KDIGO (Ngoài Dataset)",
            "actual_class": 0,
            "actual_label": "Không bệnh (Not CKD)",
            "clinical_note": "Stress-test người khỏe tập thể hình (28 tuổi, sc=1,25 cận trên do khối cơ, nhưng hemo=15,8, al=0, sg=1,020).",
            "features": {
                "age": 28.0, "bp": 75.0, "sg": "1.02", "al": "0", "su": "0",
                "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
                "bgr": 88.0, "bu": 26.0, "sc": 1.25, "sod": 141.0, "pot": 4.2,
                "hemo": 15.8, "pcv": 47.0, "wc": 6800.0, "rc": 5.3,
                "htn": "no", "dm": "no", "cad": "no", "appet": "good", "pe": "no", "ane": "no",
            },
        },
    ]
    for item in clinical_vignettes:
        item["missing_fields"] = 0
        samples.append(item)

    return samples
