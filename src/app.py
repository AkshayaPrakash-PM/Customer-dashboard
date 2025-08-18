import streamlit as st
import pandas as pd
import os
from typing import Optional


def ensure_data_dir() -> None:
    """Ensure the data directory exists."""
    os.makedirs('data', exist_ok=True)


def save_bytes_to_csv(path: str, uploaded_file) -> None:
    """Save uploaded file bytes to CSV path."""
    ensure_data_dir()
    with open(path, 'wb') as f:
        f.write(uploaded_file.getbuffer())


def read_csv_safely(path_or_buffer) -> Optional[pd.DataFrame]:
    """Read CSV safely with error handling."""
    try:
        return pd.read_csv(path_or_buffer)
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return None


def load_existing_data():
    """Load existing CSV files from data directory."""
    data_files = {
        'ocv': 'data/ocv_data.csv',
        'icms': 'data/icms_with_product.csv', 
        'customers': 'data/dummy_customers.csv'
    }
    
    loaded_data = {}
    for key, filepath in data_files.items():
        if os.path.exists(filepath):
            df = read_csv_safely(filepath)
            if df is not None:
                loaded_data[key] = df
    
    return loaded_data


def main():
    st.set_page_config(
        page_title="Customer Dashboard",
        page_icon="📊", 
        layout="wide"
    )
    
    st.title("📊 Customer Dashboard")
    st.markdown("A comprehensive dashboard for analyzing customer data from multiple sources.")
    
    # Initialize session state for data
    if 'data' not in st.session_state:
        st.session_state.data = load_existing_data()
    
    # Sidebar for file uploads
    st.sidebar.header("📁 Data Management")
    
    # Load existing data status
    st.sidebar.subheader("📈 Current Data Status")
    
    data_status_mapping = {
        'OCV Data': 'ocv',
        'ICMS with Product': 'icms',
        'Customers': 'customers'
    }
    
    for name, key in data_status_mapping.items():
        if key in st.session_state.data:
            rows = len(st.session_state.data[key])
            st.sidebar.success(f"✅ {name}: {rows:,} rows")
        else:
            st.sidebar.warning(f"⚠️ {name}: No data")
    
    # File upload section
    st.sidebar.subheader("📤 Add your CSVs")
    
    # Save to data folder checkbox
    save_to_folder = st.sidebar.checkbox(
        "Save uploads into data/ folder",
        value=False,
        help="Check this to persist uploaded files to the data/ folder for future sessions"
    )
    
    # File uploaders
    uploaded_ocv = st.sidebar.file_uploader(
        "Upload OCV data (CSV)",
        type=['csv'],
        key='ocv_uploader',
        help="Upload your OCV (Optimal Customer Value) data CSV file"
    )
    
    uploaded_icms = st.sidebar.file_uploader(
        "Upload ICMS with product (CSV)", 
        type=['csv'],
        key='icms_uploader',
        help="Upload your ICMS with product information CSV file"
    )
    
    uploaded_customers = st.sidebar.file_uploader(
        "Upload Customers (CSV)",
        type=['csv'],
        key='customers_uploader', 
        help="Upload your customer data CSV file"
    )
    
    # Process uploads
    uploads = {
        'ocv': (uploaded_ocv, 'data/ocv_data.csv'),
        'icms': (uploaded_icms, 'data/icms_with_product.csv'),
        'customers': (uploaded_customers, 'data/dummy_customers.csv')
    }
    
    for key, (uploaded_file, save_path) in uploads.items():
        if uploaded_file is not None:
            # Read the uploaded file
            df = read_csv_safely(uploaded_file)
            if df is not None:
                # Store in session state
                st.session_state.data[key] = df
                
                # Save to file if requested
                if save_to_folder:
                    try:
                        save_bytes_to_csv(save_path, uploaded_file)
                        st.sidebar.success(f"✅ {uploaded_file.name} saved to {save_path} ({len(df):,} rows)")
                    except Exception as e:
                        st.sidebar.error(f"❌ Error saving {uploaded_file.name}: {str(e)}")
                else:
                    st.sidebar.success(f"✅ {uploaded_file.name} loaded for this session ({len(df):,} rows)")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📈 OCV Data", "📦 ICMS with Product", "👥 Customers"])
    
    with tab1:
        st.header("OCV Data Analysis")
        if 'ocv' in st.session_state.data:
            df = st.session_state.data['ocv']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", f"{len(df.columns):,}")
            with col3:
                if df.select_dtypes(include=['number']).columns.any():
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    st.metric("Numeric Columns", f"{len(numeric_cols):,}")
                else:
                    st.metric("Numeric Columns", "0")
            
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(100), use_container_width=True)
            
            if not df.empty:
                st.subheader("📋 Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(col_info, use_container_width=True)
        else:
            st.info("📤 No OCV data available. Please upload a CSV file using the sidebar.")
    
    with tab2:
        st.header("ICMS with Product Data Analysis") 
        if 'icms' in st.session_state.data:
            df = st.session_state.data['icms']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", f"{len(df.columns):,}")
            with col3:
                if df.select_dtypes(include=['number']).columns.any():
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    st.metric("Numeric Columns", f"{len(numeric_cols):,}")
                else:
                    st.metric("Numeric Columns", "0")
            
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(100), use_container_width=True)
            
            if not df.empty:
                st.subheader("📋 Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(col_info, use_container_width=True)
        else:
            st.info("📤 No ICMS data available. Please upload a CSV file using the sidebar.")
    
    with tab3:
        st.header("Customer Data Analysis")
        if 'customers' in st.session_state.data:
            df = st.session_state.data['customers']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", f"{len(df.columns):,}")
            with col3:
                if df.select_dtypes(include=['number']).columns.any():
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    st.metric("Numeric Columns", f"{len(numeric_cols):,}")
                else:
                    st.metric("Numeric Columns", "0")
            
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(100), use_container_width=True)
            
            if not df.empty:
                st.subheader("📋 Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(col_info, use_container_width=True)
        else:
            st.info("📤 No customer data available. Please upload a CSV file using the sidebar.")
    
    # Footer
    st.markdown("---")
    st.markdown("📊 **Customer Dashboard** - Upload your CSV files using the sidebar to get started!")


if __name__ == "__main__":
    main()