from __future__ import annotations
import math
from datetime import datetime, timedelta
import pandas as pd
import yaml


def load_risk_config(path: str = "config/risk_weights.yml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def recent_window(df: pd.DataFrame, date_col: str, end: datetime | None = None, days: int = 90) -> pd.DataFrame:
    end = end or datetime.utcnow()
    start = end - timedelta(days=days)
    m = (df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))
    return df.loc[m].copy()


def compute_risk(ocv: pd.DataFrame, icm: pd.DataFrame, customers: pd.DataFrame, cfg: dict | None = None, as_of: datetime | None = None):
    cfg = cfg or load_risk_config()
    w = cfg.get("weights", {})
    seg_mult = cfg.get("segment_multiplier", {})
    recent_days = cfg.get("window_days", {}).get("risk_recent", 30)

    # recent slices
    ocv_recent = recent_window(ocv, "Date", as_of, days=recent_days)
    icm_recent = recent_window(icm, "Date", as_of, days=recent_days)

    # Counts for components
    open_icm = icm.groupby(["TenantID", "Severity"]).agg(
        open_count=("is_open", "sum")
    ).reset_index()
    pivot = open_icm.pivot_table(index="TenantID", columns="Severity", values="open_count", fill_value=0)
    pivot = pivot.rename(columns=lambda c: str(c).lower())
    for col in ["critical", "high", "medium"]:
        if col not in pivot.columns:
            pivot[col] = 0

    sla_breaches = icm_recent[icm_recent.get("SLA_Breach", False) == True].groupby("TenantID").size().rename("sla_breaches_30d").reset_index()

    neg_ocv = ocv_recent[ocv_recent.get("Outcome", "").eq("Negative")].groupby("TenantID").size().rename("neg_ocv_30d").reset_index()

    base = customers[["TenantID", "SalesSegment", "Licenses", "Name"]].copy()
    base = base.merge(pivot.reset_index(), how="left", left_on="TenantID", right_on="TenantID").fillna(0)
    base = base.merge(sla_breaches, how="left", on="TenantID").fillna({"sla_breaches_30d": 0})
    base = base.merge(neg_ocv, how="left", on="TenantID").fillna({"neg_ocv_30d": 0})

    base["risk_raw"] = (
        w.get("critical", 3) * base["critical"]
        + w.get("high", 2) * base["high"]
        + w.get("medium", 1) * base["medium"]
        + w.get("sla_breach_30d", 2) * base["sla_breaches_30d"]
        + w.get("neg_ocv_30d", 1) * base["neg_ocv_30d"]
    )
    base["licenses_safe"] = (base["Licenses"].fillna(0)).clip(lower=0)
    base["seg_mult"] = base["SalesSegment"].map(lambda s: seg_mult.get(str(s), 1.0))
    base["risk"] = base["risk_raw"] * (base["licenses_safe"].add(1).map(lambda x: math.log(x + 1))) * base["seg_mult"]

    base = base.sort_values("risk", ascending=False)
    return base


def compute_icm_aggregates(icm: pd.DataFrame, window_days: int = 90):
    df = recent_window(icm, "Date", days=window_days)
    agg = df.groupby(["Product", "Severity", "Status"]).size().rename("count").reset_index()
    return agg


def compute_ocv_sentiment(ocv: pd.DataFrame, window_days: int = 90):
    df = recent_window(ocv, "Date", days=window_days)
    agg = df.groupby(["Product", "Outcome"]).size().rename("count").reset_index()
    return agg


def kpi_mapping():
    # mirrors ingest constants
    return {
        "percent": {
            "Microsoft Teams": ["ActiveUsers"],
            "OneDrive": ["SyncSuccess"],
            "Exchange Online": ["DeliverySuccess"],
            "Intune": ["CompliantDevices"],
            "Defender for Endpoint": ["IncidentReduction"],
            "Entra ID": ["SSO_Coverage"],
            "Azure SQL Database": ["DTU_Utilization"],
            "Azure Virtual Desktop": ["SessionStability"],
        },
        "count": {
            "SharePoint Online": ["SitesAdopted"],
            "Power BI": ["ActiveConsumers"],
        },
    }


def kpi_trend(ocv: pd.DataFrame, product: str, metric: str, window_days: int = 90):
    df = recent_window(ocv, "Date", days=window_days)
    df = df[(df["Product"] == product) & (df["MetricName"] == metric)]
    df = df.sort_values("Date")
    return df[["Date", "TenantID", "MetricValue", "MetricUnit"]].copy()