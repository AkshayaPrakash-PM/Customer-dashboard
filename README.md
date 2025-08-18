# Customer Dashboard

A comprehensive Streamlit-based dashboard for analyzing customer data from multiple sources including OCV (Optimal Customer Value), ICMS with product information, and customer data.

## Features

- **Interactive Data Upload**: Upload CSV files directly through the web interface
- **Multiple Data Sources**: Support for three key datasets:
  - OCV data (`ocv_data.csv`)
  - ICMS with product data (`icms_with_product.csv`) 
  - Customer data (`dummy_customers.csv`)
- **Data Persistence**: Optionally save uploaded files to the `data/` folder for future sessions
- **Real-time Analysis**: Immediate data preview and analysis upon upload
- **Robust Error Handling**: Graceful handling of invalid CSV files with helpful error messages

## Quick Start

### Installation

1. Clone this repository:
```bash
git clone https://github.com/AkshayaPrakash-PM/Customer-dashboard.git
cd Customer-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run src/app.py
```

### Upload your own CSVs in the app

1. **Open the app**: Navigate to the running Streamlit application in your browser
2. **Use the sidebar "Add your CSVs"**: You'll find three file upload widgets in the left sidebar:
   - Upload OCV data (CSV)
   - Upload ICMS with product (CSV) 
   - Upload Customers (CSV)
3. **Optionally check "Save uploads into data/ folder" to persist**: 
   - If checked, your uploaded files will be saved to the `data/` directory and persist between sessions
   - If unchecked, files are only available for the current session
4. **The app will refresh and show your data in the tabs**: After successful upload, navigate through the three tabs to explore your data

## Data Structure

The dashboard expects three types of CSV data:

- **OCV Data** (`data/ocv_data.csv`): Optimal Customer Value metrics and analysis data
- **ICMS with Product** (`data/icms_with_product.csv`): Integrated Customer Management System data with product information
- **Customers** (`data/dummy_customers.csv`): Customer demographic and profile information

## Usage

### Data Upload

- Use the sidebar file uploaders to select your CSV files
- Files are validated upon upload - invalid files will show error messages
- Successfully uploaded files display row counts and confirmation messages
- The sidebar status section shows real-time data availability

### Data Analysis

Each tab provides:
- **Overview metrics**: Row count, column count, and numeric column statistics
- **Data preview**: First 100 rows of your dataset
- **Column information**: Data types, null counts, and column statistics

## Project Structure

```
Customer-dashboard/
├── src/
│   └── app.py              # Main Streamlit application
├── data/                   # CSV data files (created automatically)
│   ├── ocv_data.csv       # OCV dataset (when uploaded/saved)
│   ├── icms_with_product.csv  # ICMS dataset (when uploaded/saved)
│   └── dummy_customers.csv    # Customer dataset (when uploaded/saved)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Requirements

- Python 3.7+
- Streamlit 1.28.0+
- Pandas 2.0.0+

## Error Handling

The application includes robust error handling for:
- Invalid CSV file formats
- File read/write permissions
- Missing or corrupted data
- Network interruptions during upload

All errors are displayed with helpful messages to guide users toward resolution.