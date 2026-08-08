"""Vendor-enriched dataset normalization and feature definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import normalize_columns, require_columns, to_binary

COLUMN_ALIASES = {
    "eascore": "ea_score", "identityrank": "identity_rank",
    "devicebrowsertype": "device_browser_type",
    "ipaddressloccity": "ip_address_loc_city",
    "ipaddressloccountry": "ip_address_loc_country",
    "isvalid": "is_valid", "isconnected": "is_connected",
    "personaldevice": "personal_device", "receivingmail": "receiving_mail",
    "emaildays": "email_days", "areacode": "area_code", "opendate": "open_date",
}

NUMERIC_COLUMNS = ("ea_score", "identity_rank", "reputation_level", "volume_score", "result_number", "email_days")
BOOLEAN_COLUMNS = ("is_valid", "is_connected", "personal_device", "receiving_mail")
CATEGORICAL_COLUMNS = ("area_code", "device_browser_type", "ip_address_loc_country", "type")
TARGET_COLUMN = "is_fraud"
IDENTIFIER_COLUMNS: tuple[str, ...] = ()
EXCLUDED_AMBIGUOUS_FEATURES = ("open_date", "open_year", "open_month", "open_day_of_week")
FEATURE_METADATA = {
    "result_number": {
        "semantic_type": "numeric count",
        "source": "FraudKiller vendor",
        "meaning": "Number of results returned for the record.",
        "status": "allowed predictive feature",
    },
}
VENDOR_FEATURES = (
    "is_valid", "is_connected", "personal_device", "reputation_level",
    "receiving_mail", "type", "volume_score", "result_number", "email_days",
)
ALLOWED_PREDICTIVE_FEATURES = (*NUMERIC_COLUMNS, *BOOLEAN_COLUMNS, *CATEGORICAL_COLUMNS)
IN_HOUSE_FEATURES = tuple(column for column in ALLOWED_PREDICTIVE_FEATURES if column not in VENDOR_FEATURES)


def clean_vendor_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize names and types while preserving missing feature values.

    Missing-value imputation is deliberately deferred to model pipelines so
    validation and test observations cannot influence training statistics.
    """
    frame = normalize_columns(raw).rename(columns=COLUMN_ALIASES)
    frame = frame.replace(["NULL", "null", ""], np.nan)
    require_columns(frame, ["is_fraud"])
    if "open_date" in frame:
        frame["open_date"] = pd.to_datetime(frame["open_date"], errors="coerce")
        frame["open_year"] = frame["open_date"].dt.year
        frame["open_month"] = frame["open_date"].dt.month
        frame["open_day_of_week"] = frame["open_date"].dt.dayofweek
    frame["is_fraud"] = frame["is_fraud"].map(to_binary)
    for column in BOOLEAN_COLUMNS:
        if column in frame:
            frame[column] = frame[column].map(to_binary)
    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        if column in frame:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
    return frame
