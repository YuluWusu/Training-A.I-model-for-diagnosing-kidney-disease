from __future__ import annotations

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