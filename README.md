# Fitbit Analytics Dashboard

Transform your Google Fitbit Takeout data into detailed, organized health visualizations with higher granularity than the standard Fitbit app.

## Features

- **Continuous Data Visualization** - See every data point (second-by-second heart rate, minute-by-minute activity)
- **Higher Detail Than Fitbit App** - The Fitbit app aggregates data; this dashboard shows the raw continuous measurements
- **Interactive Charts** - Zoom, pan, and explore your health data with Plotly
- **PDF Export** - Generate printable HTML reports with embedded charts
- **Multiple Data Sources:**
  - Heart Rate (continuous, every few seconds)
  - Sleep stages and duration
  - Blood Oxygen (SpO2)
  - Heart Rate Variability (HRV)
  - Steps and Activity
  - Stress scores

## Quick Start

1. **Export your Fitbit data:**
   - Go to [Fitbit Data Export](https://www.fitbit.com/settings/data/export)
   - Request your data (may take up to 24 hours)
   - Download the ZIP file

2. **Run the dashboard:**
   ```bash
   pip install -r requirements.txt
   streamlit run health_dashboard.py
   ```

3. **Upload your data:**
   - Upload the `Takeout.zip` file in the sidebar
   - Or place your `Takeout*/Fitbit` folder in the same directory

4. **Generate PDF Report:**
   - Click "Generer rapport" in the sidebar
   - Download the HTML file
   - Open in browser and press Ctrl+P → Save as PDF

## Why This Exists

The standard Fitbit app shows daily summaries and limited history. This dashboard provides:
- **Full temporal resolution** - Every heart rate measurement, not just averages
- **Long-term trends** - View all your data at once, not day-by-day
- **Custom analysis** - Health alerts and metrics calculated from your complete dataset
- **Portable reports** - Generate PDFs for your records or healthcare provider

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- Kaleido (for PNG export)

## Screenshot

![Dashboard Screenshot](screenshot.png)

## License

MIT License
