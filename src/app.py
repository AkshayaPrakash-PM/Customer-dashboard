from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__ + "/..")))

from src.ingest import load_data, get_kpi_maps
from src.transform import compute_risk, compute_icm_aggregates, compute_ocv_sentiment, kpi_mapping, kpi_trend, load_risk_config
from src.themes import extract_themes

st.set_page_config(page_title="Customer Dashboards", layout="wide")

@st.cache_data(show_spinner=False)
def _load_all():
    ocv, icm, cust = load_data()
    return ocv, icm, cust

ocv, icm, cust = _load_all()
config = load_risk_config()

# Sidebar filters
st.sidebar.header("Filters")
end_default = ocv["Date"].max() if not ocv.empty else pd.Timestamp(datetime.utcnow().date())
start_default = end_default - pd.Timedelta(days=config.get("window_days", {}).get("default_view", 90))
start_date, end_date = st.sidebar.date_input(
    "Date range", value=(start_default.date(), end_default.date())
)
start_dt = pd.Timestamp(start_date)
end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

products = sorted(pd.unique(pd.concat([ocv["Product"], icm["Product"]], ignore_index=True).dropna()))
sel_products = st.sidebar.multiselect("Products", products, default=products)
segments = sorted(cust["SalesSegment"].dropna().unique().tolist())
sel_segments = st.sidebar.multiselect("Sales Segments", segments, default=segments)

# Apply filters
ocv_f = ocv[(ocv["Date"] >= start_dt) & (ocv["Date"] <= end_dt) & (ocv["Product"].isin(sel_products))]
icm_f = icm[(icm["Date"] >= start_dt) & (icm["Date"] <= end_dt) & (icm["Product"].isin(sel_products))]

cust_f = cust[cust["SalesSegment"].isin(sel_segments)]

st.title("Customer Dashboards (MVP)")

TAB_A, TAB_B, TAB_C = st.tabs(["Meeting Prep (S500)", "Product Review", "All Data"])

with TAB_A:
    st.subheader("Meeting Prep – prioritize S500")
    # Force S500 focus view on this tab
    s500_cust = cust[cust["SalesSegment"] == "S500"].copy()
    # Compute risk on full ICM/OCV but will show top S500
    risk_df = compute_risk(ocv, icm, cust)
    risk_s500 = risk_df[risk_df["SalesSegment"] == "S500"].copy()

    st.write("Top S500 at-risk accounts (last 90 days):")
    st.dataframe(risk_s500[["Name", "SalesSegment", "Licenses", "risk", "critical", "high", "medium", "sla_breaches_30d", "neg_ocv_30d"]].head(20))

    tenant_names = risk_s500["Name"].dropna().unique().tolist()
    sel_tenant = st.selectbox("Select tenant for prep", tenant_names)
    if sel_tenant:
        tenant_id = cust.loc[cust["Name"] == sel_tenant, "TenantID"].iloc[0]
        st.markdown(f"### {sel_tenant}")
        row = risk_s500[risk_s500["Name"] == sel_tenant].iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk score", f"{row['risk']:.1f}")
        col2.metric("Licenses", int(row["Licenses"]))
        col3.metric("Segment", row["SalesSegment"])

        # ICM status panel
        icm_t = icm_f[icm_f["TenantID"] == tenant_id]
        if not icm_t.empty:
            icm_counts = icm_t.groupby(["Severity", "Status"]).size().rename("count").reset_index()
            fig = px.bar(icm_counts, x="Severity", y="count", color="Status", barmode="group", title="ICM counts by Severity/Status")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No ICMs in selected window.")

        # OCV recent comments
        ocv_t = ocv_f[ocv_f["TenantID"] == tenant_id]
        if not ocv_t.empty:
            st.write("Recent OCV comments (last 3)")
            st.dataframe(ocv_t.sort_values("Date", ascending=False)[["Date", "Product", "Outcome", "Comment"]].head(3))
        else:
            st.info("No OCV entries in selected window.")

        # Themes
        text_df = pd.concat([
            ocv_t[["Product", "Comment"]].rename(columns={"Comment": "text"}),
            icm_t[["Product", "Complaint"]].rename(columns={"Complaint": "text"}),
        ], ignore_index=True)
        if not text_df.empty:
            themes = extract_themes(text_df, text_col="text", group_cols=["Product"], top_n=10)
            st.write("Top themes by product")
            st.dataframe(themes)

        # KPI trends per product for this tenant
        percent_map, count_map = get_kpi_maps()
        products_used = sorted(ocv_t["Product"].dropna().unique().tolist())
        for p in products_used:
            with st.expander(f"Product: {p}"):
                # Percent KPIs
                for m in percent_map.get(p, []):
                    kdf = ocv_t[(ocv_t["Product"] == p) & (ocv_t["MetricName"] == m)].sort_values("Date")
                    if not kdf.empty:
                        fig = px.line(kdf, x="Date", y="MetricValue", title=f"{m} (%)")
                        st.plotly_chart(fig, use_container_width=True)
                # Count KPIs
                for m in count_map.get(p, []):
                    kdf = ocv_t[(ocv_t["Product"] == p) & (ocv_t["MetricName"] == m)].sort_values("Date")
                    if not kdf.empty:
                        fig = px.line(kdf, x="Date", y="MetricValue", title=f"{m} (count)")
                        st.plotly_chart(fig, use_container_width=True)

with TAB_B:
    st.subheader("Product Review – big picture")
    # Aggregates
    icm_agg = compute_icm_aggregates(icm_f)
    ocv_sent = compute_ocv_sentiment(ocv_f)

    col1, col2 = st.columns(2)
    if not icm_agg.empty:
        fig1 = px.bar(icm_agg, x="Product", y="count", color="Severity", title="ICMs by Product/Severity/Status", facet_col="Status")
        col1.plotly_chart(fig1, use_container_width=True)
    if not ocv_sent.empty:
        fig2 = px.bar(ocv_sent, x="Product", y="count", color="Outcome", title="OCV sentiment by Product")
        col2.plotly_chart(fig2, use_container_width=True)

    # Sales segment mix
    if not cust_f.empty:
        seg_mix = cust_f.groupby("SalesSegment").size().rename("count").reset_index()
        fig3 = px.pie(seg_mix, names="SalesSegment", values="count", title="Customer mix by Sales Segment")
        st.plotly_chart(fig3, use_container_width=True)

    # SLA breach rate per product
    if "SLA_Breach" in icm_f.columns and not icm_f.empty:
        sla = icm_f.groupby("Product").agg(total=("SLA_Breach", "size"), breaches=("SLA_Breach", "sum")).reset_index()
        sla["breach_rate"] = (sla["breaches"] / sla["total"]).fillna(0)
        fig4 = px.bar(sla, x="Product", y="breach_rate", title="SLA breach rate by Product")
        st.plotly_chart(fig4, use_container_width=True)

    # Top themes across products
    text_df_all = pd.concat([
        ocv_f[["Product", "Comment"]].rename(columns={"Comment": "text"}),
        icm_f[["Product", "Complaint"]].rename(columns={"Complaint": "text"}),
    ], ignore_index=True)
    if not text_df_all.empty:
        themes_all = extract_themes(text_df_all, text_col="text", group_cols=["Product"], top_n=10)
        st.write("Top themes by product")
        st.dataframe(themes_all)

    # Top S500 at-risk tenants by product
    risk_all = compute_risk(ocv, icm, cust)
    risk_s500 = risk_all[risk_all["SalesSegment"] == "S500"]
    st.write("Top S500 at-risk tenants (overall)")
    st.dataframe(risk_s500[["Name", "Licenses", "risk" ]].head(20))

with TAB_C:
    st.subheader("All Data")
    st.write("OCV data")
    st.dataframe(ocv_f)
    st.write("ICM data")
    st.dataframe(icm_f)