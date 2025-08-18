# Customer Dashboards (MVP)

Streamlit dashboards that unify OCV (One Customer Voice), ICM (incidents), and customer metadata to support:
- LT Meeting Prep for an S500 customer
- Product Review big picture for LT

## Quickstart

Requirements: Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place your CSVs in data/
# - data/ocv_data.csv
# - data/icms_with_product.csv
# - data/dummy_customers.csv

streamlit run src/app.py
```

## Data assumptions
- TenantID is the join key across sources.
- ICM sentiment: Negative while Status != Closed; Neutral when Closed.
- Product aliasing: "Entra ID (Azure AD)" and "Azure AD" → "Entra ID".
- S500 customers are priority; Meeting Prep defaults to S500 segment.

## Product groups and KPIs
- Modern Work: Teams, SharePoint, OneDrive, Exchange
- Security/Identity: Defender for Endpoint, Intune, Entra ID
- Data/BI: Power BI, Azure SQL Database
- Virtualization: Azure Virtual Desktop
- Percent KPIs: ActiveUsers, SyncSuccess, DeliverySuccess, CompliantDevices, IncidentReduction, SSO_Coverage, DTU_Utilization, SessionStability
- Count KPIs: SitesAdopted (SharePoint), ActiveConsumers (Power BI)

## Config
- config/product_aliases.yml
- config/risk_weights.yml

## Notes
- The app keeps metric units separate and won't aggregate % and counts together.
- Themes are TF‑IDF based keywords (unigrams/bigrams) with English stopwords.