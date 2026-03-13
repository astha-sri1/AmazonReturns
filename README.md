# 📦 Amazon Returns Analytics Dashboard

> **North Star:** Analyze return patterns across categories, geography, time, and predict future returns using ML.

A comprehensive 9-tab Streamlit dashboard built on 3,049 Amazon return records (2020–2023).

---

## 🗂 Dashboard Tabs

| Tab | Name | Analytics Applied |
|-----|------|-------------------|
| 1 | **Overview** | KPIs, time-series trend, descriptive stats |
| 2 | **Category Analysis** | Heatmaps, stacked bars, correlation matrix |
| 3 | **Geography** | Choropleth map, K-Means geographic clustering |
| 4 | **Temporal Trends** | Seasonality heatmap, polynomial forecasting |
| 5 | **Price & Quantity** | Distributions, OLS regression, discount impact |
| 6 | **ML Model Comparison** | Linear, Decision Tree, RF, AdaBoost, GBM (R²/RMSE) |
| 7 | **Return Predictor** | RF + GBM ensemble, confidence intervals, what-if simulator |
| 8 | **High-Risk Analysis** | K-Means clusters, Z-score anomaly detection |
| 9 | **Raw Data Explorer** | Filterable table, CSV download, data quality report |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/amazon-returns-dashboard.git
cd amazon-returns-dashboard
pip install -r requirements.txt
```

### 2. Add Data

Place `Amazon_Returns.csv` in the project root directory.

### 3. Run

```bash
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`

---

## 📁 File Structure

```
amazon-returns-dashboard/
├── app.py                  # Main Streamlit dashboard (single file)
├── Amazon_Returns.csv      # Dataset (add manually)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Theme & server configuration
└── README.md
```

---

## 📊 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| OrderID | str | Unique return order ID |
| OrderDate | date | Date of return (DD-MM-YYYY) |
| CustomerID / CustomerName | str | Customer identifiers |
| ProductID / ProductName | str | Product identifiers (50 SKUs) |
| Category | str | 6 categories (Electronics, Clothing, etc.) |
| Brand | str | 10 brands |
| Quantity | int | Units returned |
| UnitPrice | float | Price per unit ($) |
| Discount | float | Discount applied (0–0.5) |
| Tax | float | Tax amount ($) |
| ShippingCost | float | Shipping cost ($) |
| TotalAmount | float | Total return value ($) — **target variable** |
| PaymentMethod | str | 6 payment methods |
| OrderStatus | str | All = "Returned" |
| City / State / Country | str | 20 cities, 5 countries |
| SellerID | str | Seller identifier |

---

## 🤖 ML Models

All models predict `TotalAmount` (return order value):

- **Linear Regression** — baseline
- **Decision Tree** — max_depth=8
- **Random Forest** — 100 trees (also used for prediction + CI)
- **AdaBoost** — 100 estimators
- **Gradient Boosting** — 100 estimators

Features: Category, Brand, PaymentMethod, City, Country (label-encoded), Quantity, UnitPrice, Discount, Tax, ShippingCost, Year, Month.

---

## 🎨 Design

Dark editorial theme with `Space Mono` + `DM Sans` + `Syne` typography, orange accent (`#f97316`), and Plotly transparent charts.

---

## 📋 Requirements

- Python 3.9+
- streamlit ≥ 1.35
- pandas, numpy, plotly, scikit-learn, statsmodels
