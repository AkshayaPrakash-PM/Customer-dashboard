import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import re

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
    """Generate comprehensive sample data for all three CSV types"""
    start_date = date.today() - timedelta(days=180)
    
    # Generate Customers Data
    segments = ["S100", "S300", "S500", "S700", "S1000"]
    regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
    products = ["Microsoft 365", "SharePoint", "Teams", "Exchange", "Azure AD", "Power BI", "OneDrive"]
    
    customers_data = []
    for i in range(1, n + 1):
        tenant_id = f"T{i:04d}"
        segment = np.random.choice(segments, p=[0.1, 0.2, 0.4, 0.2, 0.1])  # Higher probability for S500
        licenses = np.random.randint(10, 5000) if segment != "S100" else np.random.randint(1, 50)
        products_in_use = np.random.choice(products, size=np.random.randint(2, 5), replace=False)
        
        customers_data.append({
            "TenantID": tenant_id,
            "TenantName": f"Company {i}",
            "SalesSegment": segment,
            "Region": np.random.choice(regions),
            "Licenses": licenses,
            "ProductsInUse": ", ".join(products_in_use),
            "M365_MAU": int(licenses * np.random.uniform(0.6, 0.9)),  # MAU is typically 60-90% of licenses
            "AnnualRevenue": licenses * np.random.uniform(100, 300)
        })
    
    customers_df = pd.DataFrame(customers_data)
    
    # Generate ICMs Data
    severities = ["Low", "Medium", "High", "Critical"]
    statuses = ["Open", "In Progress", "Resolved", "Closed"]
    icms_data = []
    
    for _ in range(n * 3):  # 3 ICMs per customer on average
        tenant_id = f"T{np.random.randint(1, n + 1):04d}"
        severity = np.random.choice(severities, p=[0.4, 0.3, 0.2, 0.1])
        status = np.random.choice(statuses, p=[0.2, 0.3, 0.3, 0.2])
        created_date = start_date + timedelta(days=np.random.randint(0, 180))
        sla_breached = np.random.choice([True, False], p=[0.15, 0.85])
        priority_score = np.random.randint(1, 10)
        
        # Common complaint categories
        complaints = [
            "Login issues", "Performance degradation", "Sync problems", "Email delivery delays",
            "File access issues", "License allocation problems", "Integration failures",
            "Security concerns", "Data loss", "Service unavailability"
        ]
        
        icms_data.append({
            "TenantID": tenant_id,
            "Product": np.random.choice(products),
            "Severity": severity,
            "Status": status,
            "CreatedAt": created_date.strftime("%Y-%m-%d"),
            "SLA_Breached": sla_breached,
            "PriorityScore": priority_score,
            "Complaint": np.random.choice(complaints),
            "ICM_ID": f"ICM{len(icms_data) + 1:05d}"
        })
    
    icms_df = pd.DataFrame(icms_data)
    
    # Generate OCV Data
    metrics = ["Usage Hours", "Login Frequency", "Feature Adoption", "User Satisfaction", "Support Tickets"]
    outcomes = ["Positive", "Neutral", "Negative"]
    comments = [
        "Great experience overall", "Some performance issues", "Love the new features",
        "Needs improvement", "Excellent support", "Could be more intuitive",
        "Very satisfied", "Having sync issues", "Feature works well",
        "Performance is slow", "Easy to use", "Complex setup process"
    ]
    
    ocv_data = []
    for _ in range(n * 2):  # 2 OCV entries per customer on average
        tenant_id = f"T{np.random.randint(1, n + 1):04d}"
        metric = np.random.choice(metrics)
        outcome = np.random.choice(outcomes, p=[0.6, 0.25, 0.15])  # Mostly positive
        
        # Generate realistic metric values based on metric type
        if metric == "Usage Hours":
            value = np.random.randint(20, 160)
            unit = "hours/week"
        elif metric == "Login Frequency":
            value = np.random.randint(5, 50)
            unit = "logins/week"
        elif metric == "Feature Adoption":
            value = round(np.random.uniform(0.3, 0.9), 2)
            unit = "adoption_rate"
        elif metric == "User Satisfaction":
            value = round(np.random.uniform(3.0, 5.0), 1)
            unit = "rating/5"
        else:  # Support Tickets
            value = np.random.randint(0, 10)
            unit = "tickets/month"
        
        date_recorded = start_date + timedelta(days=np.random.randint(0, 180))
        
        ocv_data.append({
            "TenantID": tenant_id,
            "Product": np.random.choice(products),
            "MetricName": metric,
            "MetricValue": value,
            "MetricUnit": unit,
            "Outcome": outcome,
            "Date": date_recorded.strftime("%Y-%m-%d"),
            "Comment": np.random.choice(comments)
        })
    
    ocv_df = pd.DataFrame(ocv_data)
    
    return customers_df, icms_df, ocv_df

def join_data(customers_df, icms_df, ocv_df):
    """Join the three datasets on TenantID and perform data normalization"""
    try:
        # Ensure TenantID columns exist
        for df, name in [(customers_df, "Customers"), (icms_df, "ICMs"), (ocv_df, "OCV")]:
            if "TenantID" not in df.columns:
                raise ValueError(f"{name} data missing TenantID column")
        
        # Normalize TenantID formats (remove spaces, uppercase)
        customers_df["TenantID"] = customers_df["TenantID"].astype(str).str.strip().str.upper()
        icms_df["TenantID"] = icms_df["TenantID"].astype(str).str.strip().str.upper()
        ocv_df["TenantID"] = ocv_df["TenantID"].astype(str).str.strip().str.upper()
        
        # Convert date columns
        if "CreatedAt" in icms_df.columns:
            icms_df["CreatedAt"] = pd.to_datetime(icms_df["CreatedAt"], errors="coerce")
        if "Date" in ocv_df.columns:
            ocv_df["Date"] = pd.to_datetime(ocv_df["Date"], errors="coerce")
        
        # Convert numeric columns
        numeric_cols = ["Licenses", "M365_MAU", "PriorityScore", "MetricValue"]
        for col in numeric_cols:
            if col in customers_df.columns:
                customers_df[col] = pd.to_numeric(customers_df[col], errors="coerce")
            if col in icms_df.columns:
                icms_df[col] = pd.to_numeric(icms_df[col], errors="coerce")
            if col in ocv_df.columns:
                ocv_df[col] = pd.to_numeric(ocv_df[col], errors="coerce")
        
        return customers_df, icms_df, ocv_df
    
    except Exception as e:
        st.error(f"Error joining data: {str(e)}")
        return None, None, None

def analyze_sentiment(text):
    """Simple sentiment analysis based on keywords"""
    if pd.isna(text):
        return "Neutral"
    
    text = str(text).lower()
    positive_keywords = ["great", "excellent", "love", "satisfied", "good", "easy", "works well"]
    negative_keywords = ["issues", "problems", "slow", "complex", "improvement", "difficult", "poor"]
    
    positive_score = sum(1 for word in positive_keywords if word in text)
    negative_score = sum(1 for word in negative_keywords if word in text)
    
    if positive_score > negative_score:
        return "Positive"
    elif negative_score > positive_score:
        return "Negative"
    else:
        return "Neutral"

def get_customer_insights(tenant_id, customers_df, icms_df, ocv_df):
    """Get comprehensive insights for a specific customer"""
    # Customer basic info
    customer = customers_df[customers_df["TenantID"] == tenant_id].iloc[0] if len(customers_df[customers_df["TenantID"] == tenant_id]) > 0 else None
    
    if customer is None:
        return None
    
    # ICM analysis
    customer_icms = icms_df[icms_df["TenantID"] == tenant_id]
    open_cases = len(customer_icms[customer_icms["Status"].isin(["Open", "In Progress"])])
    sla_breaches = len(customer_icms[customer_icms["SLA_Breached"] == True])
    high_priority = len(customer_icms[customer_icms["PriorityScore"] >= 7])
    
    # Top complaints
    if "Complaint" in customer_icms.columns:
        top_complaints = customer_icms["Complaint"].value_counts().head(3).to_dict()
    else:
        top_complaints = {}
    
    # OCV analysis
    customer_ocv = ocv_df[ocv_df["TenantID"] == tenant_id]
    if len(customer_ocv) > 0 and "Comment" in customer_ocv.columns:
        sentiments = customer_ocv["Comment"].apply(analyze_sentiment).value_counts().to_dict()
    else:
        sentiments = {}
    
    # Recent feedback
    recent_feedback = customer_ocv.tail(5) if len(customer_ocv) > 0 else pd.DataFrame()
    
    return {
        "customer": customer,
        "open_cases": open_cases,
        "sla_breaches": sla_breaches,
        "high_priority": high_priority,
        "top_complaints": top_complaints,
        "sentiments": sentiments,
        "recent_feedback": recent_feedback,
        "total_icms": len(customer_icms),
        "total_ocv": len(customer_ocv)
    }

def get_product_insights(product, customers_df, icms_df, ocv_df):
    """Get comprehensive insights for a specific product"""
    # Find customers using this product
    if "ProductsInUse" in customers_df.columns:
        product_customers = customers_df[customers_df["ProductsInUse"].str.contains(product, na=False)]
    else:
        product_customers = pd.DataFrame()
    
    # ICM analysis for this product
    product_icms = icms_df[icms_df["Product"] == product] if "Product" in icms_df.columns else pd.DataFrame()
    
    # OCV analysis for this product
    product_ocv = ocv_df[ocv_df["Product"] == product] if "Product" in ocv_df.columns else pd.DataFrame()
    
    # Calculate metrics
    total_customers = len(product_customers)
    complaint_rate = len(product_icms) / total_customers if total_customers > 0 else 0
    
    # Average MAU
    avg_mau = product_customers["M365_MAU"].mean() if "M365_MAU" in product_customers.columns and len(product_customers) > 0 else 0
    
    # Resolution rate
    resolved_icms = len(product_icms[product_icms["Status"].isin(["Resolved", "Closed"])]) if len(product_icms) > 0 else 0
    resolution_rate = resolved_icms / len(product_icms) if len(product_icms) > 0 else 1.0
    
    # Top issues
    if len(product_icms) > 0 and "Complaint" in product_icms.columns:
        top_issues = product_icms["Complaint"].value_counts().head(5).to_dict()
    else:
        top_issues = {}
    
    # Sentiment analysis
    if len(product_ocv) > 0 and "Comment" in product_ocv.columns:
        sentiments = product_ocv["Comment"].apply(analyze_sentiment).value_counts().to_dict()
    else:
        sentiments = {}
    
    return {
        "total_customers": total_customers,
        "complaint_rate": complaint_rate,
        "avg_mau": avg_mau,
        "resolution_rate": resolution_rate,
        "top_issues": top_issues,
        "sentiments": sentiments,
        "customers_by_segment": product_customers["SalesSegment"].value_counts().to_dict() if len(product_customers) > 0 else {},
        "customers_by_region": product_customers["Region"].value_counts().to_dict() if len(product_customers) > 0 else {},
        "product_customers": product_customers,
        "product_icms": product_icms,
        "product_ocv": product_ocv
    }

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
    customers_df, icms_df, ocv_df = load_sample()
    st.sidebar.success("Using comprehensive sample data")
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

# Proceed with dashboard when all files are valid
if all_files_valid:
    # Join and normalize data
    customers_df, icms_df, ocv_df = join_data(customers_df, icms_df, ocv_df)
    
    if customers_df is None:
        st.error("Failed to process data. Please check your CSV files.")
        st.stop()
    
    # Main Dashboard with Tabs
    tab1, tab2 = st.tabs(["👨‍💼 Director/Customer Deep-Dive", "📊 Product Manager Overview"])
    
    with tab1:
        st.header("Director/Customer Deep-Dive View")
        
        # Customer selection
        col1, col2 = st.columns([2, 1])
        with col1:
            # Focus on S500 segment but allow all
            s500_customers = customers_df[customers_df["SalesSegment"] == "S500"]["TenantName"].tolist()
            all_customers = customers_df["TenantName"].tolist()
            
            customer_options = s500_customers + [c for c in all_customers if c not in s500_customers]
            selected_customer = st.selectbox(
                "Select Customer", 
                customer_options,
                help="S500 segment customers shown first (primary focus)"
            )
        
        with col2:
            segment_filter = st.selectbox(
                "Filter by Segment",
                ["All"] + sorted(customers_df["SalesSegment"].unique()),
                index=1 if "S500" in customers_df["SalesSegment"].unique() else 0
            )
        
        # Get selected customer data
        if selected_customer:
            selected_tenant = customers_df[customers_df["TenantName"] == selected_customer]["TenantID"].iloc[0]
            insights = get_customer_insights(selected_tenant, customers_df, icms_df, ocv_df)
            
            if insights:
                customer = insights["customer"]
                
                # Customer Profile Section
                st.subheader("📋 Customer Profile")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Customer Name", customer["TenantName"])
                    st.metric("Tenant ID", customer["TenantID"])
                
                with col2:
                    st.metric("Sales Segment", customer["SalesSegment"])
                    st.metric("Region", customer["Region"])
                
                with col3:
                    st.metric("Licenses", f"{customer['Licenses']:,}")
                    if "M365_MAU" in customer:
                        st.metric("M365 MAU", f"{customer['M365_MAU']:,}")
                
                with col4:
                    if "AnnualRevenue" in customer:
                        st.metric("Annual Revenue", f"${customer['AnnualRevenue']:,.0f}")
                    st.metric("Products in Use", len(customer["ProductsInUse"].split(", ")))
                
                # Products in Use
                st.write("**Products in Use:**", customer["ProductsInUse"])
                
                # Key Metrics Row
                st.subheader("🚨 Key Metrics & Alerts")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Open Cases", 
                        insights["open_cases"],
                        delta=-insights["open_cases"] if insights["open_cases"] > 5 else None,
                        delta_color="inverse"
                    )
                
                with col2:
                    st.metric(
                        "SLA Breaches", 
                        insights["sla_breaches"],
                        delta=-insights["sla_breaches"] if insights["sla_breaches"] > 0 else None,
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        "High Priority Cases", 
                        insights["high_priority"],
                        delta=-insights["high_priority"] if insights["high_priority"] > 2 else None,
                        delta_color="inverse"
                    )
                
                with col4:
                    total_sentiment = sum(insights["sentiments"].values())
                    positive_rate = insights["sentiments"].get("Positive", 0) / total_sentiment if total_sentiment > 0 else 0
                    st.metric(
                        "Positive Sentiment", 
                        f"{positive_rate:.1%}",
                        delta=f"{positive_rate - 0.7:.1%}" if positive_rate < 0.7 else None,
                        delta_color="normal"
                    )
                
                # Detailed Analysis Sections
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎫 Top Complaints & Pain Points")
                    if insights["top_complaints"]:
                        for complaint, count in list(insights["top_complaints"].items())[:3]:
                            st.write(f"• **{complaint}**: {count} cases")
                    else:
                        st.write("No complaints recorded")
                    
                    # Actionable Insights
                    st.subheader("💡 Actionable Insights")
                    if insights["open_cases"] > 5:
                        st.warning(f"⚠️ High number of open cases ({insights['open_cases']})")
                    if insights["sla_breaches"] > 0:
                        st.error(f"🚨 SLA breaches detected ({insights['sla_breaches']})")
                    if insights["high_priority"] > 2:
                        st.warning(f"⚡ Multiple high-priority cases ({insights['high_priority']})")
                    
                    # Positive wins
                    if positive_rate > 0.8:
                        st.success("🎉 High customer satisfaction!")
                    
                with col2:
                    st.subheader("📊 OCV Feedback & Sentiment")
                    if insights["sentiments"]:
                        sentiment_df = pd.DataFrame(list(insights["sentiments"].items()), 
                                                  columns=["Sentiment", "Count"])
                        st.bar_chart(sentiment_df.set_index("Sentiment"))
                    
                    # Recent Feedback
                    if len(insights["recent_feedback"]) > 0:
                        st.subheader("💬 Recent Feedback")
                        for _, feedback in insights["recent_feedback"].iterrows():
                            sentiment = analyze_sentiment(feedback.get("Comment", ""))
                            emoji = "🟢" if sentiment == "Positive" else "🔴" if sentiment == "Negative" else "🟡"
                            st.write(f"{emoji} {feedback.get('Comment', 'No comment')}")
    
    with tab2:
        st.header("Product Manager/Product Overview")
        
        # Product selection
        products = sorted(icms_df["Product"].unique()) if "Product" in icms_df.columns else []
        if not products:
            products = ["Microsoft 365", "SharePoint", "Teams", "Exchange", "Azure AD"]
        
        selected_product = st.selectbox("Select Product", products)
        
        if selected_product:
            product_insights = get_product_insights(selected_product, customers_df, icms_df, ocv_df)
            
            # KPI Cards
            st.subheader("📈 Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", f"{product_insights['total_customers']:,}")
            
            with col2:
                st.metric(
                    "Complaint Rate", 
                    f"{product_insights['complaint_rate']:.2f}",
                    delta=f"{product_insights['complaint_rate'] - 0.1:.2f}" if product_insights['complaint_rate'] > 0.1 else None,
                    delta_color="inverse"
                )
            
            with col3:
                st.metric("Average MAU", f"{product_insights['avg_mau']:,.0f}")
            
            with col4:
                st.metric(
                    "Resolution Rate", 
                    f"{product_insights['resolution_rate']:.1%}",
                    delta=f"{product_insights['resolution_rate'] - 0.85:.1%}" if product_insights['resolution_rate'] < 0.85 else None,
                    delta_color="normal"
                )
            
            # Customer Breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👥 Customers by Segment")
                if product_insights["customers_by_segment"]:
                    segment_df = pd.DataFrame(list(product_insights["customers_by_segment"].items()),
                                            columns=["Segment", "Count"])
                    st.bar_chart(segment_df.set_index("Segment"))
                else:
                    st.write("No customer data available")
            
            with col2:
                st.subheader("🌍 Customers by Region")
                if product_insights["customers_by_region"]:
                    region_df = pd.DataFrame(list(product_insights["customers_by_region"].items()),
                                           columns=["Region", "Count"])
                    st.bar_chart(region_df.set_index("Region"))
                else:
                    st.write("No regional data available")
            
            # Issues and Feedback Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🚨 Top Emerging Issues")
                if product_insights["top_issues"]:
                    for issue, count in list(product_insights["top_issues"].items())[:5]:
                        severity = "🔴" if count > 10 else "🟡" if count > 5 else "🟢"
                        st.write(f"{severity} **{issue}**: {count} reports")
                else:
                    st.write("No issues reported")
            
            with col2:
                st.subheader("💬 Sentiment Analysis")
                if product_insights["sentiments"]:
                    sentiment_df = pd.DataFrame(list(product_insights["sentiments"].items()),
                                              columns=["Sentiment", "Count"])
                    st.bar_chart(sentiment_df.set_index("Sentiment"))
                else:
                    st.write("No sentiment data available")
            
            # Top Customers for this Product
            st.subheader("🏆 Top Customers for " + selected_product)
            if len(product_insights["product_customers"]) > 0:
                top_customers = product_insights["product_customers"].sort_values("Licenses", ascending=False).head(10)
                
                # Make clickable for drill-down
                st.write("Click on a customer name to view their detailed profile in the Director tab:")
                
                display_customers = top_customers[["TenantName", "SalesSegment", "Region", "Licenses", "M365_MAU"]].copy()
                if "M365_MAU" not in display_customers.columns:
                    display_customers["M365_MAU"] = "N/A"
                
                st.dataframe(
                    display_customers,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.write("No customers found for this product")

else:
    st.stop()
