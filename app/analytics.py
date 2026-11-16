"""Thong ke du lieu va giai thich du doan cho website CKD.

Cac thong ke mo ta mo hinh duoc tinh tren tap Train; phan bo nhan va muc thieu
cua toan bo bo du lieu chi dung de mo ta/can doi du lieu, khong dung de fit model.
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ckd_preprocessing import (
    BINARY,
    FEATURES,
    MULTI,
    NUMERIC,
    CKDCleaner,
)

logger = logging.getLogger(__name__)

FEATURE_LABELS = {
    "age": "Tuổi",
    "bp": "Huyết áp",
    "sg": "Tỷ trọng nước tiểu",
    "al": "Albumin nước tiểu",
    "su": "Đường trong nước tiểu",
    "rbc": "Hồng cầu trong nước tiểu",
    "pc": "Tế bào mủ",
    "pcc": "Cụm tế bào mủ",
    "ba": "Vi khuẩn",
    "bgr": "Đường huyết ngẫu nhiên",
    "bu": "Urê máu",
    "sc": "Creatinine huyết thanh",
    "sod": "Natri huyết thanh",
    "pot": "Kali huyết thanh",
    "hemo": "Hemoglobin",
    "pcv": "Thể tích hồng cầu",
    "wc": "Bạch cầu",
    "rc": "Hồng cầu",
    "htn": "Tăng huyết áp",
    "dm": "Đái tháo đường",
    "cad": "Bệnh động mạch vành",
    "appet": "Tình trạng ăn uống",
    "pe": "Phù chân",
    "ane": "Thiếu máu",
}


def _json_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _clean_target(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "?": pd.NA, "nan": pd.NA})
    )


def _get_fitted_parameters(model: Any) -> dict[str, Any]:
    preprocessor = model.named_steps["preprocessor"]
    column_transformer = preprocessor.named_steps["columns"]

    numeric_pipeline = column_transformer.named_transformers_["num"]
    numeric_imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]

    binary_pipeline = column_transformer.named_transformers_["binary"]
    multi_pipeline = column_transformer.named_transformers_["multi"]

    medians = numeric_imputer.statistics_
    means = scaler.mean_
    scales = scaler.scale_
    numeric_parameters = []

    for index, feature in enumerate(NUMERIC):
        numeric_parameters.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "median": _json_number(medians[index]),
            "scaler_mean": _json_number(means[index]),
            "scaler_scale": _json_number(scales[index]),
        })

    binary_modes = binary_pipeline.named_steps["imputer"].statistics_
    multi_modes = multi_pipeline.named_steps["imputer"].statistics_

    categorical_parameters = []

    for feature, value in zip(
        BINARY + MULTI,
        list(binary_modes) + list(multi_modes)
    ):
        categorical_parameters.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "mode": str(value),
        })

    return {
        "numeric": numeric_parameters,
        "categorical": categorical_parameters,
        "input_count": len(FEATURES),
        "output_count": int(
            model.named_steps["classifier"]
            .get_booster()
            .num_features()
        ),
    }


def build_dataset_analytics(project_dir: Path, model: Any) -> dict[str, Any]:
    """Build data summary from source CSV and the already-fitted Train pipeline."""

    data_dir = project_dir / "data"

    raw_path = data_dir / "raw" / "kidney_disease.csv"
    train_path = data_dir / "processed" / "X_train_raw.csv"
    y_train_path = data_dir / "processed" / "y_train.csv"
    y_test_path = data_dir / "processed" / "y_test.csv"

    for path in (raw_path, train_path, y_train_path, y_test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy tệp dữ liệu cần thiết: {path}"
            )

    raw = pd.read_csv(raw_path)

    if (
        not set(FEATURES).issubset(raw.columns)
        or "classification" not in raw.columns
    ):
        raise ValueError(
            "Tệp kidney_disease.csv không đúng cấu trúc "
            "24 thuộc tính và nhãn classification."
        )

    # Làm sạch 24 thuộc tính bằng cùng quy tắc của mô hình.
    full_features = CKDCleaner().fit_transform(
        raw.loc[:, FEATURES]
    )

    target = _clean_target(raw["classification"])

    class_counts = {
        "not_ckd": int((target == "notckd").sum()),
        "ckd": int((target == "ckd").sum()),
    }

    # Thống kê dữ liệu thiếu trên toàn bộ file chỉ để mô tả.
    missing_by_feature = []

    for feature in FEATURES:
        count = int(full_features[feature].isna().sum())

        missing_by_feature.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "missing_count": count,
            "missing_percent": (
                count / max(len(full_features), 1) * 100
            ),
        })

    missing_by_feature.sort(
        key=lambda item: item["missing_count"],
        reverse=True
    )

    total_cells = len(full_features) * len(FEATURES)

    total_missing = int(
        full_features.isna().sum().sum()
    )

    rows_with_missing = int(
        full_features.isna().any(axis=1).sum()
    )

    rows_complete = int(
        len(full_features) - rows_with_missing
    )

    target_whitespace_count = int(
        (
            raw["classification"].astype("string") != target
        ).fillna(False).sum()
    )

    # Thống kê mô tả chỉ dùng tập Train.
    train_raw = pd.read_csv(train_path)

    train_clean = CKDCleaner().fit_transform(
        train_raw.loc[:, FEATURES]
    )

    numeric_frame = train_clean.loc[:, NUMERIC].astype(float)

    descriptions = numeric_frame.describe(
        percentiles=[0.25, 0.50, 0.75]
    ).T

    numeric_statistics = []

    for feature, row in descriptions.iterrows():
        numeric_statistics.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "count": int(row["count"]),
            "mean": _json_number(row["mean"]),
            "std": _json_number(row["std"]),
            "min": _json_number(row["min"]),
            "q1": _json_number(row["25%"]),
            "median": _json_number(row["50%"]),
            "q3": _json_number(row["75%"]),
            "max": _json_number(row["max"]),
        })

    # Tương quan Pearson được tính trên tập Train.
    corr = numeric_frame.corr(
        method="pearson",
        min_periods=30
    )

    correlations = []

    for i, left in enumerate(NUMERIC):
        for right in NUMERIC[i + 1:]:
            value = corr.loc[left, right]

            if pd.notna(value):
                pair_count = int(
                    numeric_frame[[left, right]]
                    .dropna()
                    .shape[0]
                )

                correlations.append({
                    "left": left,
                    "left_label": FEATURE_LABELS.get(left, left),
                    "right": right,
                    "right_label": FEATURE_LABELS.get(right, right),
                    "correlation": float(value),
                    "pair_count": pair_count,
                })

    correlations.sort(
        key=lambda item: abs(item["correlation"]),
        reverse=True
    )

    # Thống kê phân chia Train/Test.
    y_train = pd.read_csv(y_train_path)[
        "classification"
    ].astype(int)

    y_test = pd.read_csv(y_test_path)[
        "classification"
    ].astype(int)

    split = {
        "train_total": int(len(y_train)),
        "test_total": int(len(y_test)),
        "train_ckd": int((y_train == 1).sum()),
        "train_not_ckd": int((y_train == 0).sum()),
        "test_ckd": int((y_test == 1).sum()),
        "test_not_ckd": int((y_test == 0).sum()),
    }

    # Lấy tham số đã được Pipeline học từ Train.
    fitted = _get_fitted_parameters(model)

    age_parameters = next(
        item
        for item in fitted["numeric"]
        if item["feature"] == "age"
    )

    first_test = pd.read_csv(
        data_dir / "processed" / "X_test_raw.csv"
    ).iloc[0]

    example_age = _json_number(
        first_test.get("age")
    )

    age_z_score = None

    if example_age is not None:
        age_z_score = (
            example_age - age_parameters["scaler_mean"]
        ) / age_parameters["scaler_scale"]

    # Đọc kết quả thực nghiệm Baseline.
    baseline_rows = []

    baseline_path = (
        project_dir
        / "reports"
        / "tables"
        / ".."
        / "baseline_cv_metrics.csv"
    )

    if not baseline_path.exists():
        baseline_path = (
            project_dir
            / "reports"
            / "baseline_cv_metrics.csv"
        )

    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)

        for _, row in baseline_df.iterrows():
            baseline_rows.append({
                "model": str(row["Algorithm"]),
                "accuracy": _json_number(row["Accuracy (Mean)"]),
                "accuracy_std": _json_number(row["Accuracy (Std)"]),
                "precision": _json_number(row["Precision (Mean)"]),
                "recall": _json_number(row["Recall (Mean)"]),
                "f1": _json_number(row["F1-Score (Mean)"]),
                "roc_auc": _json_number(row["ROC-AUC (Mean)"]),
            })

    # Đọc kết quả các mô hình đã tinh chỉnh.
    tuned_rows = []

    tuned_path = (
        project_dir
        / "reports"
        / "tuned_test_metrics.csv"
    )

    if tuned_path.exists():
        tuned_df = pd.read_csv(tuned_path)

        for _, row in tuned_df.iterrows():
            tuned_rows.append({
                "model": str(row["Model"]),
                "accuracy": _json_number(row["Test Accuracy"]),
                "precision": _json_number(row["Test Precision"]),
                "recall": _json_number(row["Test Recall"]),
                "f1": _json_number(row["Test F1-Score"]),
                "roc_auc": _json_number(row["Test ROC-AUC"]),
            })

    # Đọc mức độ quan trọng của đặc trưng từ báo cáo nhóm.
    grouped_importance = []

    importance_path = (
        project_dir
        / "reports"
        / "tables"
        / "ch4_feature_importance_24_grouped.csv"
    )

    if importance_path.exists():
        importance_df = pd.read_csv(
            importance_path
        ).head(10)

        for _, row in importance_df.iterrows():
            feature = str(row["Original_Feature"])

            grouped_importance.append({
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "gain_importance": _json_number(
                    row["XGBoost_Gain_Importance"]
                ),
                "rank": int(row["Rank"]),
            })

    return {
        "dataset": {
            "name": "Chronic Kidney Disease Dataset",
            "source_file": "data/raw/kidney_disease.csv",
            "records": int(len(raw)),
            "feature_count": len(FEATURES),
            "numeric_feature_count": len(NUMERIC),
            "categorical_feature_count": len(BINARY) + len(MULTI),

            "class_distribution": class_counts,

            "class_percent": {
                "ckd": (
                    class_counts["ckd"]
                    / max(len(raw), 1) * 100
                ),
                "not_ckd": (
                    class_counts["not_ckd"]
                    / max(len(raw), 1) * 100
                ),
            },

            "label_values_normalized": target_whitespace_count,

            "total_cells": total_cells,
            "missing_cells": total_missing,
            "missing_percent": (
                total_missing / max(total_cells, 1) * 100
            ),

            "complete_rows": rows_complete,
            "rows_with_missing": rows_with_missing,
            "rows_with_missing_percent": (
                rows_with_missing / max(len(raw), 1) * 100
            ),

            "split": split,
        },

        "missing_by_feature": missing_by_feature,

        "numeric_statistics_train": numeric_statistics,

        "correlations_train": correlations[:10],

        "preprocessing": {
            **fitted,

            "zscore_example": {
                "feature": "age",
                "label": FEATURE_LABELS["age"],
                "observed_value": example_age,
                "mean_after_imputation": age_parameters["scaler_mean"],
                "scale_after_imputation": age_parameters["scaler_scale"],
                "z_score": _json_number(age_z_score),
            },

            "numeric_imputation": "Trung vị của tập Train",
            "categorical_imputation": (
                "Giá trị xuất hiện nhiều nhất của tập Train"
            ),
            "encoding": "One-Hot Encoding",
            "scaling": "StandardScaler (Z-score)",
        },

        "baseline_cv": baseline_rows,
        "tuned_test": tuned_rows,
        "feature_importance": grouped_importance,

        "method_notes": [
            "Phân bố nhãn và tỷ lệ thiếu được thống kê trên toàn bộ tệp dữ liệu sau khi chuẩn hóa ký tự.",
            "Thống kê mô tả, tương quan và tham số tiền xử lý được tính hoặc học từ tập Train.",
            "Tập Test được giữ độc lập để đánh giá mô hình; không dùng để học tham số điền khuyết hoặc chuẩn hóa.",
        ],
    }


def load_test_samples(
    project_dir: Path,
    limit_per_class: int = 5
) -> list[dict[str, Any]]:
    """Return real, deterministic examples from the saved independent Test split."""

    processed = project_dir / "data" / "processed"

    x_path = processed / "X_test_raw.csv"
    y_path = processed / "y_test.csv"
    manifest_path = processed / "split_manifest.csv"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Thiếu X_test_raw.csv hoặc y_test.csv để tạo dữ liệu mẫu."
        )

    X_test = pd.read_csv(x_path)

    y_test = pd.read_csv(y_path)[
        "classification"
    ].astype(int)

    manifest = (
        pd.read_csv(manifest_path)
        if manifest_path.exists()
        else None
    )

    test_manifest = None

    if (
        manifest is not None
        and {"split", "position", "source_row"}.issubset(
            manifest.columns
        )
    ):
        test_manifest = manifest[
            manifest["split"].astype(str).str.lower() == "test"
        ].sort_values("position")

    samples: list[dict[str, Any]] = []

    for label_value, label_name, prefix in (
        (1, "Có bệnh (CKD)", "CKD"),
        (0, "Không bệnh (Not CKD)", "NOT-CKD"),
    ):
        indices = list(
            y_test[y_test == label_value].index[:limit_per_class]
        )

        for local_number, row_index in enumerate(
            indices,
            start=1
        ):
            row = X_test.iloc[row_index]

            patient = {}

            for feature in FEATURES:
                value = row.get(feature)

                if pd.isna(value):
                    patient[feature] = None

                elif feature in MULTI:
                    patient[feature] = format(
                        float(value),
                        "g"
                    )

                elif feature in BINARY:
                    patient[feature] = (
                        str(value).strip().lower()
                    )

                else:
                    patient[feature] = float(value)

            source_row = None

            if test_manifest is not None:
                matched = test_manifest[
                    test_manifest["position"] == row_index
                ]

                if not matched.empty:
                    source_row = int(
                        matched.iloc[0]["source_row"]
                    )

            samples.append({
                "sample_code": f"{prefix}-{local_number:02d}",
                "actual_label": label_name,
                "actual_class": int(label_value),
                "split": "Test độc lập",
                "source_row": source_row,
                "missing_fields": sum(
                    value is None
                    for value in patient.values()
                ),
                "features": patient,
            })

    return samples


def build_tree_shap_calculation(
    model: Any,
    patient_df: pd.DataFrame,
    contract: dict[str, Any]
) -> dict[str, Any] | None:
    """Calculate exact TreeSHAP contributions in XGBoost raw-margin space.

    The 38 transformed columns are grouped back into the original 24 input fields.
    Positive contributions push the model score toward CKD; negative ones push it
    toward Not CKD. These explain model behavior, not medical causality.
    """

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        transformed = preprocessor.transform(patient_df)

        booster = classifier.get_booster()

        contributions = booster.predict(
            xgb.DMatrix(transformed),
            pred_contribs=True,
        )[0]

        output_features = contract.get(
            "output_features",
            []
        )

        if len(output_features) != len(contributions) - 1:
            return None

        grouped = {
            feature: 0.0
            for feature in contract["input_features"]
        }

        for transformed_name, contribution in zip(
            output_features,
            contributions[:-1]
        ):
            _, separator, tail = transformed_name.partition("__")

            if not separator:
                continue

            original_feature = tail.split("_", 1)[0]

            if original_feature in grouped:
                grouped[original_feature] += float(contribution)

        base_value = float(contributions[-1])

        raw_score = float(
            np.sum(contributions)
        )

        safe_score = max(
            min(raw_score, 700.0),
            -700.0
        )

        reconstructed_probability = (
            1.0 / (1.0 + math.exp(-safe_score))
        )

        details = []

        input_row = patient_df.iloc[0]

        for feature, impact in grouped.items():
            details.append({
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),

                "input_value": (
                    None
                    if pd.isna(input_row[feature])
                    else str(input_row[feature])
                ),

                "contribution": float(impact),

                "direction": (
                    "toward_ckd"
                    if impact > 0
                    else "toward_not_ckd"
                    if impact < 0
                    else "neutral"
                ),

                "direction_label": (
                    "Tăng điểm CKD"
                    if impact > 0
                    else "Tăng điểm Not CKD"
                    if impact < 0
                    else "Không đóng góp đáng kể"
                ),
            })

        details.sort(
            key=lambda item: abs(item["contribution"]),
            reverse=True
        )

        return {
            "method": "XGBoost TreeSHAP (pred_contribs)",

            "base_value": base_value,

            "raw_score": raw_score,

            "reconstructed_ckd_probability": (
                reconstructed_probability
            ),

            "contribution_sum": float(
                np.sum(contributions[:-1])
            ),

            "feature_count": len(details),

            "transformed_feature_count": len(
                output_features
            ),

            "contributions": details,

            "formula": (
                "F(x) = giá trị cơ sở + tổng đóng góp SHAP; "
                "P(CKD) = 1 / (1 + e^(-F(x)))."
            ),

            "explanation": (
                "Giá trị SHAP dương làm tăng điểm dự đoán CKD; "
                "giá trị âm làm tăng điểm dự đoán Not CKD."
            ),

            "limitation": (
                "SHAP giải thích mức đóng góp của đặc trưng vào dự đoán "
                "của mô hình, không chứng minh quan hệ nhân quả "
                "hoặc thay thế kết luận y khoa."
            ),
        }

    except Exception:
        logger.exception(
            "Không thể tính toán phần giải thích TreeSHAP cho dữ liệu đầu vào."
        )
        return None
