import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

np.random.seed(7)
st.set_page_config(page_title="Customer Dashboard", layout="wide")

REQUIRED_FIELDS = ["customer_id", "segment", "region", "revenue", "date"]

def _guess_col(df_cols, candidates):
    cols_lc = {c.lower(): c for c in df_cols}
    # exact matches
    for cand in candidates:
        if cand in cols_lc:
            return cols_lc[cand]
    # partial matches
    for c in df_cols:
        lc = c.lower()
        if any(tok in lc for tok in candidates):
            return c
    return None

def guess_mapping(df):
    mapping = {}
    mapping["customer_id"] = _guess_col(df.columns, ["customer_id", "customer id", "id", "cust_id", "cid"])
    mapping["segment"] = _guess_col(df.columns, ["segment", "customer_segment", "category", "tier"])
    mapping["region"]  = _guess_col(df.columns, ["region", "area", "territory", "state"])
    mapping["revenue"] = _guess_col(df.columns, ["revenue", "amount", "sales", "value", "net_sales"])
    mapping["date"]    = _guess_col(df.columns, ["date", "order_date", "invoice_date", "created_at", "timestamp"])
    return mapping

@st.cache_data
def load_sample(n=200):
    start = pd.Timestamp.today() - pd.Timedelta(days=180)
    segments = ["Enterprise", "Mid‑market", "SMB"]
    regions = ["North", "South", "East", "West"]
    data = []
    for i in range(1, n + 1):
        data.append({
            "customer_id": f"C{i:04d}",
            "segment": np.random.choice(segments),
            "region": np.random.choice(regions),
            "revenue": float(np.random.gamma(4, 250)),
            "date": start + pd.Timedelta(days=int(np.random.randint(0, 180)))
        })
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])  # normalize to pandas datetime
    return df

def normalize_df(df, mapping):
    # Rename to standard names
    renamed = {}
    for std_name, src_col in mapping.items():
        if src_col:
            renamed[src_col] = std_name
    df = df.rename(columns=renamed)

    # Ensure required columns exist
    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    # Coerce types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["date", "revenue"])  # drop rows with bad dates or revenue
    return df

st.title("Customer Dashboard")

with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose data source", ["Upload CSV", "Use sample data"], index=0)

df = None
load_error = None

if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"]) 
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            st.sidebar.success(f"Loaded {len(raw):,} rows")
            st.sidebar.caption("Map your columns to the expected fields")
            guessed = guess_mapping(raw)

            with st.sidebar.expander("Map columns", expanded=True):
                cols = list(raw.columns)
                def idx_of(x):
                    return cols.index(x) if x in cols else 0
                c_id      = st.selectbox("Customer ID column", cols, index=idx_of(guessed["customer_id"]))
                c_segment = st.selectbox("Segment column", cols, index=idx_of(guessed["segment"]))
                c_region  = st.selectbox("Region column", cols, index=idx_of(guessed["region"]))
                c_revenue = st.selectbox("Revenue column", cols, index=idx_of(guessed["revenue"]))
                c_date    = st.selectbox("Date column", cols, index=idx_of(guessed["date"]))

            mapping = {
                "customer_id": c_id,
                "segment": c_segment,
                "region": c_region,
                "revenue": c_revenue,
                "date": c_date,
            }
            df = normalize_df(raw.copy(), mapping)
        except Exception as e:
            load_error = str(e)
else:
    df = load_sample()

if load_error:
    st.error(f"Could not load your CSV: {load_error}")
    st.stop()

if df is None or df.empty:
    st.info("Upload a CSV in the sidebar to get started, or switch to 'Use sample data'.")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    seg = st.multiselect("Segment", sorted(df["segment"].dropna().unique()), default=list(sorted(df["segment"].dropna().unique())))
    reg = st.multiselect("Region", sorted(df["region"].dropna().unique()), default=list(sorted(df["region"].dropna().unique())))
    min_rev = st.slider("Minimum revenue", 0, int(max(0, df["revenue"].max())) + 1, 0, step=50)
    default_start = pd.to_datetime(df["date"]).min().date()
    default_end = pd.to_datetime(df["date"]).max().date()
    date_range = st.date_input("Date range", (default_start, default_end))

# Normalize date filter bounds
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start, default_end
start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date)

# Apply filters
f = df[
    df["segment"].isin(seg) &
    df["region"].isin(reg) &
    (df["revenue"] >= min_rev) &
    (df["date"].between(start_ts, end_ts))
]

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Total Revenue", f"${f['revenue'].sum():,.0f}")
c2.metric("Avg Revenue/Customer", f"${(f['revenue'].mean() if len(f) else 0):,.0f}")
c3.metric("Customers", f"{f['customer_id'].nunique():,}")

# Charts
cc1, cc2 = st.columns(2)
with cc1:
    st.subheader("Revenue by Segment")
    st.bar_chart(f.groupby("segment", as_index=False)["revenue"].sum(), x="segment", y="revenue", height=300)
with cc2:
    st.subheader("Revenue by Region")
    st.bar_chart(f.groupby("region", as_index=False)["revenue"].sum(), x="region", y="revenue", height=300)

st.subheader("Customers")
st.dataframe(
    f.sort_values("revenue", ascending=False).reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)

st.caption("Tip: Use the sidebar to upload your CSV and map columns. Date must be a date/time column; Revenue must be numeric.")
