# Test CSV Upload Functionality

This document describes how to test the three CSV upload functionality implemented in this milestone.

## Test Files Created

The following test CSV files have been created in the `/tmp` directory for testing:

### 1. Valid CSV Files

- `/tmp/sample_customers.csv` - Valid customers data with all required columns
- `/tmp/sample_icms.csv` - Valid ICMs data with all required columns  
- `/tmp/sample_ocv.csv` - Valid OCV data with all required columns

### 2. Invalid CSV Files

- `/tmp/invalid_customers.csv` - Missing required columns (Licenses, ProductsInUse)

## How to Test

1. **Start the application**: `streamlit run app.py`
2. **Navigate to**: http://localhost:8501
3. **Test scenarios**:

### Scenario 1: No files uploaded
- Expected: Info message asking to upload all three CSV files
- ✅ Verified: Shows "Please upload the following CSV files to continue: Customers, ICMs, OCV"

### Scenario 2: Sample data fallback
- Check "Use sample data (fallback)" checkbox
- Expected: Legacy dashboard with customer data appears
- ✅ Verified: Dashboard shows with filters, charts, and data table

### Scenario 3: Valid CSV upload (manual test)
- Upload the three valid CSV files from `/tmp/`
- Expected: Success messages, data preview sections, and success page
- Expected output: "🎉 All three CSV files uploaded successfully!"

### Scenario 4: Invalid CSV upload (manual test)
- Upload `/tmp/invalid_customers.csv` as the customers file
- Expected: Error message showing missing columns
- Expected: "✗ Customers CSV missing required columns: SalesSegment, Licenses, ProductsInUse"

## Validation Logic Tested

The following validation functions have been verified to work correctly:

```python
# Valid customers CSV: True, Customers CSV valid
# Invalid customers CSV: False, Customers CSV missing required columns: SalesSegment, Licenses, ProductsInUse
# Valid ICMs CSV: True, ICMs CSV valid
# Valid OCV CSV: True, OCV CSV valid
```

## Required Columns

### Customers CSV
- TenantID, TenantName, SalesSegment, Licenses, ProductsInUse

### ICMs CSV  
- TenantID, Product, Severity, Status, CreatedAt, SLA_Breached, PriorityScore

### OCV CSV
- TenantID, Product, MetricName, MetricValue, MetricUnit, Outcome, Date, Comment

## UI Features Implemented

- ✅ Three separate file upload widgets with clear labels
- ✅ Help text showing required columns for each CSV
- ✅ Real-time validation with success/error messages
- ✅ Row count display for valid uploads
- ✅ Expandable data preview sections
- ✅ Sample data fallback checkbox
- ✅ Blocking logic - dashboard only proceeds when all files valid
- ✅ Clean error messaging for missing files or invalid columns

## Next Milestone

The next milestone will implement:
- Data joining on TenantID
- Risk scoring algorithm
- Dashboard views (Meeting Prep, Product Review)
- Product aliasing and theme extraction