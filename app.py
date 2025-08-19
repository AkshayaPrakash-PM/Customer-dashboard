import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

np.random.seed(7)
st.set_page_config(page_title="Customer Dashboard", layout="wide")

# Required fields for each CSV type based on issue #6
REQUIRED_FIELDS_CUSTOMERS = ["TenantID", "TenantName", "SalesSegment", "Licenses", "ProductsInUse"]
REQUIRED_FIELDS_ICMS = ["TenantID", "Product", "Severity", "Status", "CreatedAt", "SLA_Breached", "PriorityScore"]
REQUIRED_FIELDS_OCV = ["TenantID", "Product", "MetricName", "MetricValue", "MetricUnit", "Outcome", "Date", "Comment"]

# Legacy required fields for sample data fallback
REQUIRED_FIELDS = ["customer_id", "segment", "region", "revenue", "date"]

def validate_csv(df, required_fields, csv_type):
    """Validate that a CSV has all required columns"""
    if df is None or df.empty:
        return False, f"{csv_type} CSV is empty"
    
    missing_cols = [col for col in required_fields if col not in df.columns]
    if missing_cols:
        return False, f"{csv_type} CSV missing required columns: {', '.join(missing_cols)}"
    
    return True, f"{csv_type} CSV valid"

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
    start = date.today() - timedelta(days=180)
    segments = ["Enterprise", "Mid‑market", "SMB"]
    regions = ["North", "South", "East", "West"]
    data = []
    for i in range(1, n + 1):
        data.append({
            "customer_id": f"C{i:04d}",
            "segment": np.random.choice(segments),
            "region": np.random.choice(regions),
            "revenue": float(np.random.gamma(4, 250)),
            "date": start + timedelta(days=int(np.random.randint(0, 180)))
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
    st.header("Data Upload")
    st.caption("Upload three CSV files to get started")
    
    # File uploaders for the three required CSVs
    customers_file = st.file_uploader("Customers CSV", type=["csv"], key="customers", 
                                     help="Required columns: TenantID, TenantName, SalesSegment, Licenses, ProductsInUse")
    icms_file = st.file_uploader("ICMs CSV", type=["csv"], key="icms",
                                help="Required columns: TenantID, Product, Severity, Status, CreatedAt, SLA_Breached, PriorityScore")
    ocv_file = st.file_uploader("OCV CSV", type=["csv"], key="ocv",
                               help="Required columns: TenantID, Product, MetricName, MetricValue, MetricUnit, Outcome, Date, Comment")
    
    st.divider()
    use_sample = st.checkbox("Use sample data (fallback)", value=False)

# Initialize variables
customers_df = None
icms_df = None
ocv_df = None
load_errors = []
all_files_valid = False

if use_sample:
    # Fallback to sample data
    customers_df = load_sample()
    st.sidebar.success("Using sample customer data")
    all_files_valid = True
else:
    # Process uploaded files
    uploaded_files = []
    
    if customers_file is not None:
        try:
            customers_df = pd.read_csv(customers_file)
            is_valid, message = validate_csv(customers_df, REQUIRED_FIELDS_CUSTOMERS, "Customers")
            if is_valid:
                st.sidebar.success(f"✓ Customers: {len(customers_df):,} rows")
                uploaded_files.append("customers")
            else:
                st.sidebar.error(f"✗ {message}")
                load_errors.append(message)
                customers_df = None
        except Exception as e:
            error_msg = f"Could not load Customers CSV: {str(e)}"
            st.sidebar.error(f"✗ {error_msg}")
            load_errors.append(error_msg)
            customers_df = None
    
    if icms_file is not None:
        try:
            icms_df = pd.read_csv(icms_file)
            is_valid, message = validate_csv(icms_df, REQUIRED_FIELDS_ICMS, "ICMs")
            if is_valid:
                st.sidebar.success(f"✓ ICMs: {len(icms_df):,} rows")
                uploaded_files.append("icms")
            else:
                st.sidebar.error(f"✗ {message}")
                load_errors.append(message)
                icms_df = None
        except Exception as e:
            error_msg = f"Could not load ICMs CSV: {str(e)}"
            st.sidebar.error(f"✗ {error_msg}")
            load_errors.append(error_msg)
            icms_df = None
    
    if ocv_file is not None:
        try:
            ocv_df = pd.read_csv(ocv_file)
            is_valid, message = validate_csv(ocv_df, REQUIRED_FIELDS_OCV, "OCV")
            if is_valid:
                st.sidebar.success(f"✓ OCV: {len(ocv_df):,} rows")
                uploaded_files.append("ocv")
            else:
                st.sidebar.error(f"✗ {message}")
                load_errors.append(message)
                ocv_df = None
        except Exception as e:
            error_msg = f"Could not load OCV CSV: {str(e)}"
            st.sidebar.error(f"✗ {error_msg}")
            load_errors.append(error_msg)
            ocv_df = None
    
    # Check if all three files are uploaded and valid
    all_files_valid = len(uploaded_files) == 3 and customers_df is not None and icms_df is not None and ocv_df is not None

# Display upload status
if not use_sample:
    if load_errors:
        st.error("Please fix the following issues with your CSV files:")
        for error in load_errors:
            st.write(f"• {error}")
    
    if customers_file is None or icms_file is None or ocv_file is None:
        missing = []
        if customers_file is None:
            missing.append("Customers")
        if icms_file is None:
            missing.append("ICMs") 
        if ocv_file is None:
            missing.append("OCV")
        st.info(f"Please upload the following CSV files to continue: {', '.join(missing)}")
    
    if not all_files_valid:
        st.stop()

# Data preview sections
if all_files_valid and not use_sample:
    st.subheader("Data Preview")
    
    with st.expander("👥 Customers Data", expanded=False):
        st.write(f"**{len(customers_df):,} rows** with columns: {', '.join(customers_df.columns)}")
        st.dataframe(customers_df.head(10), use_container_width=True)
    
    with st.expander("🎫 ICMs Data", expanded=False):
        st.write(f"**{len(icms_df):,} rows** with columns: {', '.join(icms_df.columns)}")
        st.dataframe(icms_df.head(10), use_container_width=True)
    
    with st.expander("📊 OCV Data", expanded=False):
        st.write(f"**{len(ocv_df):,} rows** with columns: {', '.join(ocv_df.columns)}")
        st.dataframe(ocv_df.head(10), use_container_width=True)

# For now, continue with sample data logic if using fallback
if use_sample:
    df = customers_df  # This maintains compatibility with existing dashboard code
elif all_files_valid:
    # Placeholder for future implementation - show success message
    st.success("🎉 All three CSV files uploaded successfully!")
    st.info("📈 Dashboard functionality for the three-CSV data structure will be implemented in the next milestone.")
    st.write("**Next steps:** Data joining on TenantID, risk scoring, and dashboard views will be added.")
    st.stop()
else:
    st.stop()

# Ensure df is available for legacy dashboard code when using sample data
if use_sample and df is not None:
    # Continue with existing dashboard logic
    pass
else:
    # This shouldn't happen if logic above is correct
    st.error("Unable to load data")
    st.stop()

# Sidebar filters - only show when using sample data
if use_sample and df is not None:
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
