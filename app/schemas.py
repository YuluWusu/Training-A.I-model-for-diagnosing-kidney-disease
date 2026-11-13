
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Nhom chi so sinh hoa mau va huyet hoc
    age: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    bp: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    # Nhom chi so nuoc tieu
    sg: Literal[
        "1.005", "1.01", "1.015", "1.02", "1.025"
    ] | None = None

    al: Literal["0", "1", "2", "3", "4", "5"] | None = None
    su: Literal["0", "1", "2", "3", "4", "5"] | None = None

    rbc: Literal["normal", "abnormal"] | None = None
    pc: Literal["normal", "abnormal"] | None = None
    pcc: Literal["notpresent", "present"] | None = None
    ba: Literal["notpresent", "present"] | None = None

    # Nhom chi so sinh hoa mau con lai
    bgr: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    bu: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    sc: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    sod: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    pot: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    hemo: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    pcv: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    wc: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    rc: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    # Nhom tien su benh ly va lam sang
    htn: Literal["no", "yes"] | None = None
    dm: Literal["no", "yes"] | None = None
    cad: Literal["no", "yes"] | None = None
    appet: Literal["good", "poor"] | None = None
    pe: Literal["no", "yes"] | None = None
    ane: Literal["no", "yes"] | None = None
