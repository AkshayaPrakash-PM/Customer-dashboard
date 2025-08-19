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
    # Implement the dashboard functionality with the uploaded CSV data
    # Join the data on TenantID and create dashboard views
    
    # Data preparation and joining
    @st.cache_data
    def prepare_dashboard_data(customers_df, icms_df, ocv_df):
        """Join and prepare data for dashboard views"""
        # Ensure TenantID is string type for consistent joining
        customers_df['TenantID'] = customers_df['TenantID'].astype(str)
        icms_df['TenantID'] = icms_df['TenantID'].astype(str)
        ocv_df['TenantID'] = ocv_df['TenantID'].astype(str)
        
        # Convert date columns
        if 'CreatedAt' in icms_df.columns:
            icms_df['CreatedAt'] = pd.to_datetime(icms_df['CreatedAt'], errors='coerce')
        if 'Date' in ocv_df.columns:
            ocv_df['Date'] = pd.to_datetime(ocv_df['Date'], errors='coerce')
            
        return customers_df, icms_df, ocv_df
    
    customers_df, icms_df, ocv_df = prepare_dashboard_data(customers_df, icms_df, ocv_df)
    
    # Create tabs for the two dashboard views
    tab1, tab2 = st.tabs(["🎯 Director/Customer Deep-Dive", "📊 Product Manager Overview"])
    
    with tab1:
        st.header("Director/Customer Deep-Dive Dashboard")
        st.caption("Focus on S500 customers and deep customer insights")
        
        # Customer selection with S500 focus
        col1, col2 = st.columns([2, 1])
        with col1:
            # Filter for S500 customers by default
            s500_customers = customers_df[customers_df['SalesSegment'] == 'S500']['TenantName'].tolist()
            all_customers = customers_df['TenantName'].tolist()
            
            default_customers = s500_customers if s500_customers else all_customers[:5]
            selected_customer = st.selectbox(
                "Select Customer (S500 customers shown first)",
                options=s500_customers + [c for c in all_customers if c not in s500_customers],
                index=0 if s500_customers else 0
            )
        
        with col2:
            segment_filter = st.selectbox("Filter by Segment", 
                                        options=['All'] + list(customers_df['SalesSegment'].unique()),
                                        index=0)
        
        # Get selected customer data
        if selected_customer:
            customer_data = customers_df[customers_df['TenantName'] == selected_customer].iloc[0]
            tenant_id = customer_data['TenantID']
            
            # Customer Profile Section
            st.subheader(f"📋 Customer Profile: {selected_customer}")
            
            profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)
            with profile_col1:
                st.metric("Segment", customer_data['SalesSegment'])
                st.metric("TenantID", tenant_id)
            with profile_col2:
                st.metric("Licenses", f"{customer_data['Licenses']:,}")
                if 'Region' in customer_data:
                    st.metric("Region", customer_data['Region'])
            with profile_col3:
                st.metric("Products in Use", len(customer_data['ProductsInUse'].split(',')))
                if 'M365_MAU' in customer_data:
                    st.metric("M365 MAU", f"{customer_data['M365_MAU']:,}")
            with profile_col4:
                # Calculate key metrics
                customer_icms = icms_df[icms_df['TenantID'] == tenant_id]
                open_cases = len(customer_icms[customer_icms['Status'] == 'Open'])
                sla_breaches = len(customer_icms[customer_icms['SLA_Breached'] == True])
                st.metric("Open Cases", open_cases)
                st.metric("SLA Breaches", sla_breaches)
            
            # Products in Use
            st.write("**Products in Use:**", customer_data['ProductsInUse'])
            
            # ICMs and Issues Section
            st.subheader("🎫 Issues & Support Cases")
            
            if not customer_icms.empty:
                # ICM Summary
                icm_col1, icm_col2, icm_col3 = st.columns(3)
                with icm_col1:
                    critical_high = len(customer_icms[customer_icms['Severity'].isin(['Critical', 'High'])])
                    st.metric("Critical/High Severity", critical_high, 
                             help="Number of critical and high severity issues")
                with icm_col2:
                    avg_priority = customer_icms['PriorityScore'].mean()
                    st.metric("Avg Priority Score", f"{avg_priority:.1f}",
                             help="Average priority score across all issues")
                with icm_col3:
                    urgent_issues = len(customer_icms[(customer_icms['SLA_Breached'] == True) | 
                                                    (customer_icms['PriorityScore'] > 80)])
                    st.metric("Urgent Issues", urgent_issues,
                             help="SLA breached or priority score > 80")
                
                # Top Issues Table
                st.write("**Recent Issues:**")
                issues_display = customer_icms.sort_values('PriorityScore', ascending=False)[
                    ['Product', 'Severity', 'Status', 'CreatedAt', 'SLA_Breached', 'PriorityScore']
                ].head(10)
                st.dataframe(issues_display, use_container_width=True, hide_index=True)
                
                # Top 3 Pain Points
                if len(customer_icms) >= 3:
                    st.write("**🔥 Top 3 Pain Points:**")
                    top_issues = customer_icms.nlargest(3, 'PriorityScore')
                    for i, (_, issue) in enumerate(top_issues.iterrows(), 1):
                        severity_color = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
                        st.write(f"{i}. {severity_color.get(issue['Severity'], '⚪')} **{issue['Product']}** - {issue['Severity']} severity (Priority: {issue['PriorityScore']})")
            else:
                st.info("No support cases found for this customer.")
            
            # OCV Feedback Section
            st.subheader("💬 Customer Voice & Feedback")
            customer_ocv = ocv_df[ocv_df['TenantID'] == tenant_id]
            
            if not customer_ocv.empty:
                # Sentiment Analysis
                sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
                
                sentiment_counts = customer_ocv['Outcome'].value_counts()
                with sentiment_col1:
                    positive = sentiment_counts.get('Positive', 0)
                    st.metric("Positive Feedback", positive, 
                             help="Number of positive feedback entries")
                with sentiment_col2:
                    neutral = sentiment_counts.get('Neutral', 0) 
                    st.metric("Neutral Feedback", neutral,
                             help="Number of neutral feedback entries")
                with sentiment_col3:
                    negative = sentiment_counts.get('Negative', 0)
                    st.metric("Negative Feedback", negative,
                             help="Number of negative feedback entries")
                
                # Recent Feedback
                st.write("**Recent Feedback:**")
                recent_feedback = customer_ocv.sort_values('Date', ascending=False)[
                    ['Product', 'MetricName', 'MetricValue', 'Outcome', 'Date', 'Comment']
                ].head(5)
                st.dataframe(recent_feedback, use_container_width=True, hide_index=True)
                
                # Actionable Insights
                st.subheader("💡 Actionable Insights")
                insights = []
                
                # Check for unresolved complaints
                if urgent_issues > 0:
                    insights.append(f"🚨 **{urgent_issues} urgent issues** require immediate attention")
                
                # Check for positive wins
                if positive > negative:
                    insights.append(f"✅ **Strong customer satisfaction** with {positive} positive vs {negative} negative feedback")
                
                # Check for escalation needs
                if sla_breaches > 0:
                    insights.append(f"⚠️ **{sla_breaches} SLA breaches** - escalation recommended")
                
                # Usage insights
                if 'M365_MAU' in customer_data and customer_data['M365_MAU'] < customer_data['Licenses'] * 0.7:
                    usage_rate = (customer_data['M365_MAU'] / customer_data['Licenses']) * 100
                    insights.append(f"📈 **Low usage rate** ({usage_rate:.1f}%) - opportunity for adoption programs")
                
                if insights:
                    for insight in insights:
                        st.write(insight)
                else:
                    st.write("✅ No immediate action items identified")
                    
            else:
                st.info("No customer feedback data available for this customer.")
    
    with tab2:
        st.header("Product Manager Overview Dashboard")
        st.caption("Product-focused insights and customer analytics")
        
        # Product selection
        all_products = set()
        for products_str in customers_df['ProductsInUse'].dropna():
            all_products.update([p.strip() for p in products_str.split(',')])
        all_products = sorted(list(all_products))
        
        selected_product = st.selectbox("Select Product", options=all_products)
        
        if selected_product:
            # Filter data for selected product
            product_customers = customers_df[customers_df['ProductsInUse'].str.contains(selected_product, na=False)]
            product_icms = icms_df[icms_df['Product'] == selected_product]
            product_ocv = ocv_df[ocv_df['Product'] == selected_product]
            
            # KPI Cards
            st.subheader(f"📊 {selected_product} - Key Metrics")
            
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                total_customers = len(product_customers)
                st.metric("Total Customers", total_customers)
            with kpi_col2:
                if len(product_icms) > 0 and total_customers > 0:
                    complaint_rate = (len(product_icms) / total_customers) * 100
                    st.metric("Complaint Rate", f"{complaint_rate:.1f}%")
                else:
                    st.metric("Complaint Rate", "0%")
            with kpi_col3:
                if 'M365_MAU' in product_customers.columns:
                    avg_mau = product_customers['M365_MAU'].mean()
                    st.metric("Average MAU", f"{avg_mau:,.0f}")
                else:
                    st.metric("Average MAU", "N/A")
            with kpi_col4:
                if len(product_icms) > 0:
                    resolution_rate = (len(product_icms[product_icms['Status'].isin(['Resolved', 'Closed'])]) / len(product_icms)) * 100
                    st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
                else:
                    st.metric("Resolution Rate", "100%")
            
            # Customer Breakdown
            st.subheader("👥 Customer Breakdown")
            
            breakdown_col1, breakdown_col2 = st.columns(2)
            
            with breakdown_col1:
                st.write("**By Segment:**")
                segment_breakdown = product_customers['SalesSegment'].value_counts()
                if not segment_breakdown.empty:
                    segment_df = pd.DataFrame({
                        'Segment': segment_breakdown.index,
                        'Customers': segment_breakdown.values
                    })
                    st.bar_chart(segment_df.set_index('Segment'))
                    
                    # Show numbers
                    for segment, count in segment_breakdown.items():
                        percentage = (count / total_customers) * 100
                        st.write(f"• **{segment}**: {count} customers ({percentage:.1f}%)")
                else:
                    st.write("No customer data available")
            
            with breakdown_col2:
                st.write("**By Region:**")
                if 'Region' in product_customers.columns:
                    region_breakdown = product_customers['Region'].value_counts()
                    if not region_breakdown.empty:
                        region_df = pd.DataFrame({
                            'Region': region_breakdown.index,
                            'Customers': region_breakdown.values
                        })
                        st.bar_chart(region_df.set_index('Region'))
                        
                        # Show numbers
                        for region, count in region_breakdown.items():
                            percentage = (count / total_customers) * 100
                            st.write(f"• **{region}**: {count} customers ({percentage:.1f}%)")
                    else:
                        st.write("No region data available")
                else:
                    st.write("Region data not available")
            
            # Top Emerging Issues
            st.subheader("🚨 Top Emerging Issues")
            
            if not product_icms.empty:
                # Group by severity and count
                severity_issues = product_icms.groupby(['Severity', 'Status']).size().reset_index(name='Count')
                
                issues_col1, issues_col2 = st.columns(2)
                
                with issues_col1:
                    st.write("**By Severity:**")
                    severity_counts = product_icms['Severity'].value_counts()
                    for severity, count in severity_counts.items():
                        severity_color = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
                        st.write(f"{severity_color.get(severity, '⚪')} **{severity}**: {count} issues")
                
                with issues_col2:
                    st.write("**Recent High-Priority Issues:**")
                    recent_high_priority = product_icms[
                        (product_icms['Severity'].isin(['Critical', 'High'])) |
                        (product_icms['PriorityScore'] > 75)
                    ].sort_values(['PriorityScore', 'CreatedAt'], ascending=[False, False]).head(5)
                    
                    if not recent_high_priority.empty:
                        for _, issue in recent_high_priority.iterrows():
                            customer_name = customers_df[customers_df['TenantID'] == issue['TenantID']]['TenantName'].iloc[0] if len(customers_df[customers_df['TenantID'] == issue['TenantID']]) > 0 else issue['TenantID']
                            st.write(f"• **{customer_name}** - {issue['Severity']} (Priority: {issue['PriorityScore']})")
                    else:
                        st.write("No high-priority issues")
            else:
                st.info("No issues reported for this product")
            
            # OCV Feedback Analysis
            st.subheader("💬 Customer Voice Analysis")
            
            if not product_ocv.empty:
                feedback_col1, feedback_col2 = st.columns(2)
                
                with feedback_col1:
                    st.write("**Sentiment Distribution:**")
                    sentiment_dist = product_ocv['Outcome'].value_counts()
                    sentiment_df = pd.DataFrame({
                        'Sentiment': sentiment_dist.index,
                        'Count': sentiment_dist.values
                    })
                    st.bar_chart(sentiment_df.set_index('Sentiment'))
                
                with feedback_col2:
                    st.write("**Recent Feedback Highlights:**")
                    recent_feedback = product_ocv.sort_values('Date', ascending=False).head(5)
                    for _, feedback in recent_feedback.iterrows():
                        outcome_emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
                        customer_name = customers_df[customers_df['TenantID'] == feedback['TenantID']]['TenantName'].iloc[0] if len(customers_df[customers_df['TenantID'] == feedback['TenantID']]) > 0 else feedback['TenantID'] 
                        st.write(f"{outcome_emoji.get(feedback['Outcome'], '💬')} **{customer_name}**: {feedback['Comment'][:100]}...")
            else:
                st.info("No customer feedback available for this product")
            
            # Top Customers for this Product
            st.subheader("🏆 Top Customers")
            
            if not product_customers.empty:
                # Sort by licenses or MAU if available
                if 'M365_MAU' in product_customers.columns:
                    top_customers = product_customers.nlargest(10, 'M365_MAU')[
                        ['TenantName', 'SalesSegment', 'Licenses', 'M365_MAU']
                    ]
                else:
                    top_customers = product_customers.nlargest(10, 'Licenses')[
                        ['TenantName', 'SalesSegment', 'Licenses']
                    ]
                
                st.dataframe(top_customers, use_container_width=True, hide_index=True)
            else:
                st.info("No customers found for this product")
                
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
