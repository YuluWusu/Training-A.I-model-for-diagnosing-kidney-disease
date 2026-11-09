"""Tiền xử lý CKD — Huỳnh Công Tạo. Module phải đi cùng file joblib."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc',
            'sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']
NUMERIC = ['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
BINARY = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
MULTI = ['sg','al','su']
BINARY_LEVELS = [['normal','abnormal'], ['normal','abnormal'],
                 ['notpresent','present'], ['notpresent','present'],
                 ['no','yes'], ['no','yes'], ['no','yes'],
                 ['good','poor'], ['no','yes'], ['no','yes']]
MULTI_LEVELS = [['1.005','1.01','1.015','1.02','1.025'],
                ['0','1','2','3','4','5'], ['0','1','2','3','4','5']]
MISSING = {'', '?', 'nan', 'none', 'null', 'na', 'n/a'}

def normalize_text(series):
    """Quy tắc cố định, không học thống kê từ dữ liệu."""
    return series.map(lambda v: np.nan if pd.isna(v) or str(v).strip().lower() in MISSING
                      else str(v).strip().lower()).astype(object)

class CKDCleaner(TransformerMixin, BaseEstimator):
    """Chuẩn hóa 24 cột; từ chối cột lạ/thiếu và giá trị không hợp lệ."""
    def fit(self, X, y=None):
        self.transform(X)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Đầu vào phải là pandas DataFrame có tên 24 cột.')
        if X.columns.duplicated().any() or set(X.columns) != set(FEATURES):
            raise ValueError(f'Cần đúng 24 cột. Thiếu={set(FEATURES)-set(X.columns)}, '
                             f'thừa={set(X.columns)-set(FEATURES)}')
        out = X.loc[:, FEATURES].copy()
        for col in FEATURES:
            out[col] = normalize_text(out[col])
        for col in NUMERIC + MULTI:
            val = pd.to_numeric(out[col], errors='coerce')
            invalid = out[col].notna() & val.isna()
            if invalid.any():
                raise ValueError(f'{col}: giá trị không phải số {out.loc[invalid,col].unique()}')
            if ((val.notna()) & (~np.isfinite(val) | (val < 0))).any():
                raise ValueError(f'{col}: có giá trị âm hoặc vô hạn.')
            out[col] = val.astype(float)
        for col in MULTI:
            out[col] = out[col].map(lambda v: np.nan if pd.isna(v) else format(v, 'g')).astype(object)
        for col, levels in zip(BINARY + MULTI, BINARY_LEVELS + MULTI_LEVELS):
            invalid = out[col].notna() & ~out[col].isin(levels)
            if invalid.any():
                raise ValueError(f'{col}: ngoài miền cho phép: {out.loc[invalid,col].unique()}')
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)

def make_preprocessor():
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
    ])
    binary = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('encoder', OneHotEncoder(categories=BINARY_LEVELS, drop='if_binary',
                                  handle_unknown='error', sparse_output=False)),
    ])
    multi = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('encoder', OneHotEncoder(categories=MULTI_LEVELS, drop=None,
                                  handle_unknown='error', sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ('num', numeric, NUMERIC), ('binary', binary, BINARY), ('multi', multi, MULTI),
    ], remainder='drop', verbose_feature_names_out=True)
    return Pipeline([('clean', CKDCleaner()), ('columns', columns)])
