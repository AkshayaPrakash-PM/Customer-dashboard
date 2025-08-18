import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
np.random.seed(7)

st.set_page_config(page_title="Customer Dashboard", layout="wide")

@st.cache_data
def load_data(n=200):
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
            "date": start + timedelta(days=int(np.random.randint(0, 180)))  # Python date objects
        })
    df = pd.DataFrame(data)
    # Normalize to pandas datetime so comparisons work
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

st.title("Customer Dashboard")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    seg = st.multiselect("Segment", sorted(df["segment"].unique()), default=list(df["segment"].unique()))
    reg = st.multiselect("Region", sorted(df["region"].unique()), default=list(df["region"].unique()))
    min_rev = st.slider("Minimum revenue", 0, int(df["revenue"].max()) + 1, 0, step=50)

    # date_input wants date objects; use .date() for the defaults
    default_start = df["date"].min().date()
    default_end = df["date"].max().date()
    date_range = st.date_input("Date range", (default_start, default_end))

# Unpack and normalize bounds to pandas Timestamps
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = df["date"].min().date(), df["date"].max().date()

start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date)

# Apply filters
f = df[
    df["segment"].isin(seg) &
    df["region"].isin(reg) &
    (df["revenue"] >= min_rev) &
    (df["date"].between(start_ts, end_ts))
]

# Top KPIs
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

st.caption("Demo data generated locally. Replace with your real data when ready.")