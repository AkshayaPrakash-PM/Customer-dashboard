from __future__ import annotations
import os
import pandas as pd
import numpy as np
from dateutil import parser
import yaml

PRODUCT_GROUPS = {
    "Microsoft Teams": "Modern Work",
    "SharePoint Online": "Modern Work",
    "OneDrive": "Modern Work",
    "Exchange Online": "Modern Work",
    "Defender for Endpoint": "Security/Identity",
    "Intune": "Security/Identity",
    "Entra ID": "Security/Identity",
    "Entra ID (Azure AD)": "Security/Identity",  # will be normalized
    "Power BI": "Data/BI",
    "Azure SQL Database": "Data/BI",
    "Azure Virtual Desktop": "Virtualization",
}

PERCENT_KPIS = {
    "Microsoft Teams": ["ActiveUsers"],
    "OneDrive": ["SyncSuccess"],
    "Exchange Online": ["DeliverySuccess"],
    "Intune": ["CompliantDevices"],
    "Defender for Endpoint": ["IncidentReduction"],
    "Entra ID": ["SSO_Coverage"],
    "Azure SQL Database": ["DTU_Utilization"],
    "Azure Virtual Desktop": ["SessionStability"],
}
COUNT_KPIS = {
    "SharePoint Online": ["SitesAdopted"],
    "Power BI": ["ActiveConsumers"],
}


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column headers (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    return df


def _parse_dates(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df["year"] = df[col].dt.year
        df["month"] = df[col].dt.to_period("M").astype(str)
        df["week"] = df[col].dt.to_period("W").astype(str)
    return df


def _load_aliases(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return y.get("aliases", {})


def _apply_aliases(df: pd.DataFrame, product_col: str, aliases: dict) -> pd.DataFrame:
    if product_col in df.columns:
        df[product_col] = df[product_col].map(lambda x: aliases.get(str(x), str(x)))
    return df


def _add_product_group(df: pd.DataFrame, product_col: str = "Product") -> pd.DataFrame:
    if product_col in df.columns:
        df["product_group"] = df[product_col].map(lambda x: PRODUCT_GROUPS.get(str(x), "Other"))
    return df


def load_data(data_dir: str = "data", aliases_path: str = "config/product_aliases.yml"):
    ocv = _read_csv(os.path.join(data_dir, "ocv_data.csv"))
    icm = _read_csv(os.path.join(data_dir, "icms_with_product.csv"))
    cust = _read_csv(os.path.join(data_dir, "dummy_customers.csv"))

    # Normalize customers: unify Sales Segment header -> SalesSegment
    if "Sales Segment" in cust.columns and "SalesSegment" not in cust.columns:
        cust = cust.rename(columns={"Sales Segment": "SalesSegment"})

    # Apply aliases
    aliases = _load_aliases(aliases_path)
    ocv = _apply_aliases(ocv, "Product", aliases)
    icm = _apply_aliases(icm, "Product", aliases)

    # Parse dates
    ocv = _parse_dates(ocv, "Date")
    icm = _parse_dates(icm, "Date")

    # Types
    if "MetricValue" in ocv.columns:
        ocv["MetricValue"] = pd.to_numeric(ocv["MetricValue"], errors="coerce")

    # Add product groups
    ocv = _add_product_group(ocv, "Product")
    icm = _add_product_group(icm, "Product")

    # ICM: simple sentiment Option A
    if set(["Status", "Severity"]).issubset(icm.columns):
        icm["is_open"] = icm["Status"].fillna("") != "Closed"
        icm["icm_sentiment_label"] = np.where(icm["is_open"], "Negative", "Neutral")

    # Join customers
    if "TenantID" in ocv.columns:
        # Drop existing SalesSegment if it exists to avoid conflicts
        ocv_cols_to_drop = [col for col in ocv.columns if col in ["Name", "SalesSegment", "Licenses"]]
        if ocv_cols_to_drop:
            ocv = ocv.drop(columns=ocv_cols_to_drop)
        ocv = ocv.merge(cust[["TenantID", "Name", "SalesSegment", "Licenses"]], on="TenantID", how="left")
    if "TenantID" in icm.columns:
        # Drop existing SalesSegment if it exists to avoid conflicts
        icm_cols_to_drop = [col for col in icm.columns if col in ["Name", "SalesSegment", "Licenses"]]
        if icm_cols_to_drop:
            icm = icm.drop(columns=icm_cols_to_drop)
        icm = icm.merge(cust[["TenantID", "Name", "SalesSegment", "Licenses"]], on="TenantID", how="left")
    
    return ocv, icm, cust


def get_kpi_maps():
    return PERCENT_KPIS, COUNT_KPIS


def get_product_groups():
    return PRODUCT_GROUPS

if __name__ == "__main__":
    ocv, icm, cust = load_data()
    print(ocv.head())
    print(icm.head())