# Customer Dashboard

A comprehensive Streamlit dashboard for customer management and product analytics, featuring two specialized views for different user personas.

## Features

### 🔄 CSV Upload & Data Integration
- **Three CSV Upload System**: Upload Customers, ICMs (Incident Management), and OCV (Outcome Validation) data
- **Data Joining**: Automatic joining on TenantID with robust error handling
- **Data Validation**: Real-time validation of required columns for each CSV type
- **Sample Data Fallback**: Comprehensive sample data for testing and demonstration

### 👨‍💼 Director/Customer Deep-Dive View
- **Customer Selection**: Search and select customers with focus on S500 segment
- **Customer Profile**: Complete customer information including:
  - Name, Segment, TenantID, Region
  - Licenses, ProductsInUse, M365 MAU
  - Annual Revenue and utilization metrics
- **Issue Management**: Track open cases, SLA breaches, and high-priority incidents
- **Sentiment Analysis**: OCV feedback analysis with positive/negative sentiment tracking
- **Actionable Insights**: Automated alerts for escalation needs and positive wins
- **Historical Context**: Recent feedback and trend analysis

### 📊 Product Manager/Product Overview View
- **Product Analytics**: Select any product for comprehensive analysis
- **Customer Breakdown**: Segmentation by region, sales segment, and license volume
- **Issue Tracking**: Top emerging complaints with severity indicators
- **KPI Dashboard**: Key metrics including:
  - Total customers per product
  - Complaint rate and resolution rate
  - Average MAU (Monthly Active Users)
- **Sentiment Analysis**: Product-specific feedback and sentiment trends
- **Drill-Down**: Navigate from product view to individual customer details

### 🎨 User Interface
- **Tab-Based Layout**: Easy switching between Director and Product Manager views
- **Interactive Charts**: Visual representations of data trends and distributions
- **Metric Cards**: Real-time KPIs with delta indicators
- **Responsive Design**: Optimized for different screen sizes
- **Actionable Alerts**: Color-coded warnings and success indicators

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Access the Dashboard**:
   - Open your browser to `http://localhost:8501`
   - Use sample data for immediate testing, or upload your own CSVs

## CSV Data Requirements

### Customers CSV
Required columns: `TenantID`, `TenantName`, `SalesSegment`, `Licenses`, `ProductsInUse`

### ICMs CSV (Incident Management)
Required columns: `TenantID`, `Product`, `Severity`, `Status`, `CreatedAt`, `SLA_Breached`, `PriorityScore`

### OCV CSV (Outcome Validation)
Required columns: `TenantID`, `Product`, `MetricName`, `MetricValue`, `MetricUnit`, `Outcome`, `Date`, `Comment`

## Usage

1. **Upload Data**: Use the sidebar to upload three CSV files or enable sample data
2. **Director View**: Select customers and analyze their health, issues, and feedback
3. **Product Manager View**: Analyze product performance, customer distribution, and emerging issues
4. **Drill-Down**: Use insights from either view to dive deeper into specific areas

## Technical Details

- **Framework**: Streamlit for web interface
- **Data Processing**: Pandas for data manipulation and analysis
- **Visualization**: Built-in Streamlit charts and metrics
- **Data Validation**: Comprehensive CSV validation with error reporting
- **Sentiment Analysis**: Keyword-based sentiment classification
- **Error Handling**: Robust error handling throughout the application