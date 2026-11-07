"""The fields a rule may read. Anything not listed here is rejected by the type checker."""

from dataclasses import dataclass
from enum import Enum


class FieldType(Enum):
    NUMBER = "number"
    INT = "int"
    BOOL = "bool"
    ENUM = "enum"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: FieldType
    min: float | None = None
    max: float | None = None
    values: tuple[str, ...] = ()


METHODS = ("UPI", "CARD", "NETBANKING", "WALLET")
CATEGORIES = ("GROCERY", "FOOD", "TRAVEL", "GAMING", "ELECTRONICS", "UTILITIES")

_SPECS = (
    FieldSpec("txn.amount", FieldType.NUMBER, 0, 10_000_000),
    FieldSpec("txn.method", FieldType.ENUM, values=METHODS),
    FieldSpec("txn.hour", FieldType.INT, 0, 23),
    FieldSpec("txn.is_international", FieldType.BOOL),
    FieldSpec("txn.count_1h", FieldType.INT, 0, 1000),
    FieldSpec("device.age_days", FieldType.INT, 0, 10_000),
    FieldSpec("device.is_rooted", FieldType.BOOL),
    FieldSpec("merchant.category", FieldType.ENUM, values=CATEGORIES),
    FieldSpec("merchant.whitelisted", FieldType.BOOL),
    FieldSpec("user.account_age_days", FieldType.INT, 0, 10_000),
    FieldSpec("user.failed_attempts_24h", FieldType.INT, 0, 100),
    FieldSpec("user.kyc_verified", FieldType.BOOL),
)

SCHEMA: dict[str, FieldSpec] = {spec.name: spec for spec in _SPECS}
