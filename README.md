# Customer Dashboard (Scaffold)

This is a minimal starter so you can run the Streamlit app and see the tabs. It works even if the data files are missing.

## Quick start

### Option 1 — Download ZIP (no Git needed)
1. Click the green "Code" button → "Download ZIP".
2. Unzip it. Open the folder called `Customer-dashboard`.
3. Put your CSV files into the `data/` folder:
   - `data/ocv_data.csv`
   - `data/icms_with_product.csv`
   - `data/dummy_customers.csv`
4. Open Terminal (macOS/Linux) or Command Prompt (Windows) inside the folder.
5. Run these commands:

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run src/app.py
   ```

   Windows:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run src/app.py
   ```

6. Your browser opens to http://localhost:8501

### Option 2 — Git clone
```bash
git clone https://github.com/AkshayaPrakash-PM/Customer-dashboard.git
cd Customer-dashboard
python -m venv .venv
# or python3 on macOS/Linux
. .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run src/app.py
```

## Notes
- If the CSV files aren't there yet, the app still runs and shows what to add.
- After you add the files, click the "Rerun" button at the top-right (or press R) to refresh.
- To stop the app, return to the terminal and press `Ctrl + C`.