import os
from typing import Tuple, Dict

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Dashboard", layout="wide")

REQUIRED_FILES = {
    "ocv": "data/ocv_data.csv",
    "icms": "data/icms_with_product.csv",
    "customers": "data/dummy_customers.csv",
}


def files_status() -> Dict[str, bool]:
    return {k: os.path.exists(v) for k, v in REQUIRED_FILES.items()}


def load_data() -> Tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    status = files_status()
    def _read(path: str) -> pd.DataFrame | None:
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Could not read {path}: {e}")
            return None
    ocv = _read(REQUIRED_FILES["ocv"]) if status["ocv"] else None
    icms = _read(REQUIRED_FILES["icms"]) if status["icms"] else None
    customers = _read(REQUIRED_FILES["customers"]) if status["customers"] else None
    return ocv, icms, customers


def show_missing_files_help():
    st.info(
        """
        To see full dashboards, add the following CSV files into the data/ folder:
        - data/ocv_data.csv
        - data/icms_with_product.csv
        - data/dummy_customers.csv

        After adding files, click the "Rerun" button at the top-right (or press R).
        """
    )


def main():
    st.title("Customer Dashboard")
    st.caption("Minimal scaffold: runs even if data is missing. Add CSVs to data/ to unlock full views.")

    # Sidebar status
    st.sidebar.header("Data files status")
    status = files_status()
    for key, path in REQUIRED_FILES.items():
        present = status[key]
        st.sidebar.write(f"{'✅' if present else '❌'} {path}")

    ocv, icms, customers = load_data()

    tabs = st.tabs(["Meeting Prep (S500)", "Product Review", "All Data"])

    with tabs[0]:
        st.subheader("Meeting Prep (S500)")
        if ocv is None or customers is None:
            show_missing_files_help()
        else:
            st.write("OCV sample (first 10 rows):")
            st.dataframe(ocv.head(10), use_container_width=True)
            st.write("Customers sample:")
            st.dataframe(customers.head(10), use_container_width=True)
            st.success("Data loaded. Replace with your analysis/visuals as needed.")

    with tabs[1]:
        st.subheader("Product Review")
        if icms is None:
            show_missing_files_help()
        else:
            st.write("ICMS sample (first 10 rows):")
            st.dataframe(icms.head(10), use_container_width=True)
            # Example simple KPI
            st.write("Rows in ICMS:", len(icms))

    with tabs[2]:
        st.subheader("All Data")
        if ocv is not None:
            st.write("OCV:")
            st.dataframe(ocv, use_container_width=True)
        else:
            st.write("OCV data not found.")
        if icms is not None:
            st.write("ICMS:")
            st.dataframe(icms, use_container_width=True)
        else:
            st.write("ICMS data not found.")
        if customers is not None:
            st.write("Customers:")
            st.dataframe(customers, use_container_width=True)
        else:
            st.write("Customers data not found.")


if __name__ == "__main__":
    main()