"""
Amazon Returns Dashboard — app.py
North Star: Analysis of Return Patterns across Categories, Geography, Time & ML Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                               GradientBoostingRegressor, RandomForestClassifier)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Amazon Returns Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — dark, data-dense, editorial aesthetic
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

:root {
    --bg:       #0d0f14;
    --surface:  #151821;
    --border:   #252a35;
    --accent:   #f97316;
    --accent2:  #6366f1;
    --accent3:  #22d3ee;
    --text:     #e8eaf0;
    --muted:    #8b92a5;
    --success:  #10b981;
    --danger:   #ef4444;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif !important;
    color: var(--accent) !important;
    letter-spacing: 0.05em;
}

/* KPI cards */
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 16px 20px;
    text-align: left;
}
.kpi-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1;
}
.kpi-delta {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent3);
    margin-top: 4px;
}

/* Section headers */
.tab-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text);
    border-bottom: 2px solid var(--accent);
    padding-bottom: 6px;
    margin-bottom: 4px;
    letter-spacing: 0.02em;
}
.tab-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 20px;
}

/* Metric pill */
.pill {
    display: inline-block;
    background: rgba(249,115,22,0.15);
    color: var(--accent);
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: 999px;
    padding: 2px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
}

/* Tables */
.styled-table {
    width:100%; border-collapse:collapse; font-size:0.82rem;
}
.styled-table th {
    background:var(--border); color:var(--accent);
    font-family:'Space Mono',monospace; font-size:0.7rem;
    letter-spacing:0.08em; text-transform:uppercase;
    padding:8px 12px; text-align:left;
}
.styled-table td {
    padding:7px 12px; border-bottom:1px solid var(--border);
    color:var(--text);
}
.styled-table tr:hover td { background:var(--border); }

/* Plotly chart background */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* Streamlit overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--surface);
    border-radius: 8px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    border-radius: 6px;
    padding: 6px 12px;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
}
div[data-testid="stMetric"] label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted) !important;
    letter-spacing: 0.1em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    color: var(--text) !important;
}
/* Dataframe */
.stDataFrame { border: 1px solid var(--border); border-radius: 6px; }

/* Selectbox / multiselect */
.stMultiSelect [data-baseweb="tag"] { background: var(--accent2) !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def process_data(raw_bytes: bytes):
    import io
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], dayfirst=True)
    df["Year"]      = df["OrderDate"].dt.year
    df["Month"]     = df["OrderDate"].dt.month
    df["MonthName"] = df["OrderDate"].dt.strftime("%b")
    df["YearMonth"] = df["OrderDate"].dt.to_period("M").astype(str)
    df["Quarter"]   = df["OrderDate"].dt.to_period("Q").astype(str)
    df["DayOfWeek"] = df["OrderDate"].dt.day_name()
    df["NetRevenue"] = df["TotalAmount"] - df["ShippingCost"] - df["Tax"]
    return df

# ── Resolve data source: local file → sidebar uploader ────────────────────────
import os, pathlib

_LOCAL_PATHS = ["Amazon_Returns.csv", "data/Amazon_Returns.csv"]
_raw_bytes = None

for _p in _LOCAL_PATHS:
    if pathlib.Path(_p).exists():
        _raw_bytes = pathlib.Path(_p).read_bytes()
        break

if _raw_bytes is None:
    # Show a clean upload prompt and stop execution until the file is provided
    st.markdown("""
    <div style='text-align:center;padding:60px 20px'>
        <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#f97316'>
            📦 Amazon Returns Dashboard
        </div>
        <div style='font-family:DM Sans,sans-serif;color:#8b92a5;margin-top:8px;font-size:1rem'>
            Upload your <code>Amazon_Returns.csv</code> to launch the dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Amazon_Returns.csv",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("👆 Please upload **Amazon_Returns.csv** to continue.")
        st.stop()

    _raw_bytes = uploaded.read()

df_full = process_data(_raw_bytes)

# ── Plotly theme helper ────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#e8eaf0", size=12),
    legend=dict(bgcolor="rgba(21,24,33,0.8)", bordercolor="#252a35", borderwidth=1),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=["#f97316","#6366f1","#22d3ee","#10b981","#f43f5e","#a78bfa","#fbbf24"],
)
def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#252a35", zeroline=False)
    fig.update_yaxes(gridcolor="#252a35", zeroline=False)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — GLOBAL FILTERS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📦 AMAZON RETURNS")
    st.markdown("<div style='font-family:Space Mono;font-size:0.65rem;color:#8b92a5;letter-spacing:0.1em;margin-bottom:16px'>RETURN PATTERN DASHBOARD</div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**Global Filters**")

    all_cats   = sorted(df_full["Category"].unique())
    all_brands = sorted(df_full["Brand"].unique())
    all_countries = sorted(df_full["Country"].unique())

    sel_cats = st.multiselect("Category", all_cats, default=all_cats, key="g_cat")
    sel_brands = st.multiselect("Brand", all_brands, default=all_brands, key="g_brand")
    sel_countries = st.multiselect("Country", all_countries, default=all_countries, key="g_country")

    date_min = df_full["OrderDate"].min().date()
    date_max = df_full["OrderDate"].max().date()
    date_range = st.date_input("Date Range", value=(date_min, date_max),
                               min_value=date_min, max_value=date_max)

    st.divider()
    st.markdown("<div style='font-family:Space Mono;font-size:0.62rem;color:#8b92a5'>ALL ORDERS = RETURNED STATUS<br>3,049 records · 2020–2023</div>", unsafe_allow_html=True)

# Apply global filters
if len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    d0, d1 = df_full["OrderDate"].min(), df_full["OrderDate"].max()

df = df_full[
    df_full["Category"].isin(sel_cats) &
    df_full["Brand"].isin(sel_brands) &
    df_full["Country"].isin(sel_countries) &
    (df_full["OrderDate"] >= d0) &
    (df_full["OrderDate"] <= d1)
].copy()

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Overview",
    "🗂 Category Analysis",
    "🌍 Geography",
    "📅 Temporal Trends",
    "💰 Price & Quantity",
    "🤖 ML Model Comparison",
    "🔮 Return Predictor",
    "⚠️ High-Risk Analysis",
    "🔎 Raw Data Explorer",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="tab-header">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Descriptive statistics · Time-series return trends</div>', unsafe_allow_html=True)

    # KPIs
    total_returns  = len(df)
    total_revenue  = df["TotalAmount"].sum()
    avg_amount     = df["TotalAmount"].mean()
    avg_qty        = df["Quantity"].mean()
    avg_price      = df["UnitPrice"].mean()
    total_shipping = df["ShippingCost"].sum()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpis = [
        (c1, "Total Returns",       f"{total_returns:,}",         "all orders returned"),
        (c2, "Total Revenue Lost",  f"${total_revenue:,.0f}",     "sum of TotalAmount"),
        (c3, "Avg Return Value",    f"${avg_amount:,.2f}",        "mean TotalAmount"),
        (c4, "Avg Quantity",        f"{avg_qty:.2f}",             "items per return"),
        (c5, "Avg Unit Price",      f"${avg_price:.2f}",          "avg UnitPrice"),
        (c6, "Total Shipping Cost", f"${total_shipping:,.0f}",    "logistics burden"),
    ]
    for col, label, value, sub in kpis:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Time trend — returns per month
    col_l, col_r = st.columns([2,1])
    with col_l:
        trend = df.groupby("YearMonth").agg(Returns=("OrderID","count"), Revenue=("TotalAmount","sum")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=trend["YearMonth"], y=trend["Returns"],
                             name="Returns", marker_color="#f97316", opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=trend["YearMonth"], y=trend["Revenue"],
                                 name="Revenue ($)", line=dict(color="#22d3ee", width=2),
                                 mode="lines+markers", marker=dict(size=4)), secondary_y=True)
        fig.update_layout(title="Monthly Returns & Revenue", **PLOTLY_LAYOUT)
        fig.update_xaxes(gridcolor="#252a35", tickangle=-45)
        fig.update_yaxes(gridcolor="#252a35")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Category breakdown donut
        cat_cnt = df["Category"].value_counts().reset_index()
        cat_cnt.columns = ["Category","Count"]
        fig2 = px.pie(cat_cnt, names="Category", values="Count",
                      hole=0.55, title="Returns by Category",
                      color_discrete_sequence=["#f97316","#6366f1","#22d3ee","#10b981","#f43f5e","#a78bfa"])
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # Descriptive stats table
    st.markdown("**Descriptive Statistics**")
    desc = df[["Quantity","UnitPrice","Discount","Tax","ShippingCost","TotalAmount","NetRevenue"]].describe().T.round(2)
    st.dataframe(desc, use_container_width=True)

    # Day-of-week heatmap
    col_a, col_b = st.columns(2)
    with col_a:
        dow = df.groupby("DayOfWeek").size().reindex(
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index()
        dow.columns = ["Day","Returns"]
        fig3 = px.bar(dow, x="Day", y="Returns", title="Returns by Day of Week",
                      color="Returns", color_continuous_scale="Oranges")
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        yearly = df.groupby("Year").agg(Returns=("OrderID","count"),
                                         AvgValue=("TotalAmount","mean")).reset_index()
        fig4 = px.bar(yearly, x="Year", y="Returns", text="Returns",
                      title="Returns by Year", color_discrete_sequence=["#6366f1"])
        fig4.update_traces(textposition="outside")
        apply_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CATEGORY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="tab-header">Category Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Descriptive stats · Visual heatmaps · Category vs TotalAmount correlation</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3,1])
    with c1:
        cat_brand = df.groupby(["Category","Brand"]).agg(
            Returns=("OrderID","count"), Revenue=("TotalAmount","sum")).reset_index()
        fig = px.bar(cat_brand, x="Category", y="Returns", color="Brand",
                     title="Returns by Category & Brand (Stacked)",
                     barmode="stack",
                     color_discrete_sequence=["#f97316","#6366f1","#22d3ee","#10b981","#f43f5e","#a78bfa","#fbbf24","#fb7185","#34d399","#818cf8"])
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cat_agg = df.groupby("Category")["TotalAmount"].mean().reset_index()
        fig2 = px.pie(cat_agg, names="Category", values="TotalAmount",
                      title="Avg Return Value Share", hole=0.5,
                      color_discrete_sequence=["#f97316","#6366f1","#22d3ee","#10b981","#f43f5e","#a78bfa"])
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap: Category × Brand → avg TotalAmount
    pivot = df.pivot_table(values="TotalAmount", index="Category",
                           columns="Brand", aggfunc="mean").fillna(0).round(0)
    fig3 = px.imshow(pivot, text_auto=True, aspect="auto",
                     color_continuous_scale="Oranges",
                     title="Heatmap: Avg Return Value (Category × Brand)")
    apply_theme(fig3)
    st.plotly_chart(fig3, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        # Box plot — TotalAmount by Category
        fig4 = px.box(df, x="Category", y="TotalAmount", color="Category",
                      title="Return Value Distribution by Category",
                      color_discrete_sequence=["#f97316","#6366f1","#22d3ee","#10b981","#f43f5e","#a78bfa"])
        apply_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)
    with col_r:
        # Correlation: category encoded vs TotalAmount
        le = LabelEncoder()
        df_enc = df.copy()
        df_enc["Category_enc"] = le.fit_transform(df_enc["Category"])
        corr_data = df_enc[["Category_enc","Quantity","UnitPrice","Discount","Tax","ShippingCost","TotalAmount"]].corr()
        fig5 = px.imshow(corr_data, text_auto=".2f", color_continuous_scale="RdBu_r",
                         title="Correlation Matrix (Category + Numerics)")
        apply_theme(fig5)
        st.plotly_chart(fig5, use_container_width=True)

    # Brand-level summary table
    st.markdown("**Brand Performance Summary**")
    brand_sum = df.groupby("Brand").agg(
        Returns=("OrderID","count"),
        TotalRevenue=("TotalAmount","sum"),
        AvgValue=("TotalAmount","mean"),
        AvgDiscount=("Discount","mean"),
        AvgQty=("Quantity","mean")
    ).round(2).sort_values("Returns", ascending=False).reset_index()
    st.dataframe(brand_sum, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GEOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="tab-header">Geography</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Descriptive stats · Geographic clustering · Choropleth map</div>', unsafe_allow_html=True)

    country_filter = st.selectbox("Focus Country", ["All"] + sorted(df["Country"].unique()))
    df_geo = df if country_filter == "All" else df[df["Country"] == country_filter]

    # Country choropleth
    country_cnt = df_geo.groupby("Country").agg(
        Returns=("OrderID","count"), AvgValue=("TotalAmount","mean")).reset_index()
    fig = px.choropleth(country_cnt, locations="Country", locationmode="country names",
                        color="Returns", hover_data=["AvgValue"],
                        color_continuous_scale="Oranges",
                        title="Returns by Country (Choropleth)")
    apply_theme(fig)
    fig.update_geos(bgcolor="rgba(0,0,0,0)", showframe=False,
                    landcolor="#252a35", oceancolor="#0d0f14",
                    showcoastlines=True, coastlinecolor="#3a3f50")
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        # Top cities
        city_agg = df_geo.groupby(["City","Country","State"]).agg(
            Returns=("OrderID","count"), AvgValue=("TotalAmount","mean")).reset_index()
        city_agg = city_agg.sort_values("Returns", ascending=False).head(20)
        fig2 = px.bar(city_agg, x="Returns", y="City", orientation="h",
                      color="Returns", color_continuous_scale="Oranges",
                      title="Top 20 Cities by Return Volume", text="Returns")
        fig2.update_traces(textposition="outside")
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        state_agg = df_geo.groupby("State").agg(
            Returns=("OrderID","count"), AvgValue=("TotalAmount","mean")).reset_index()
        fig3 = px.bar(state_agg.sort_values("Returns",ascending=False),
                      x="State", y="Returns",
                      color="AvgValue", color_continuous_scale="Viridis",
                      title="Returns by State (colored by Avg Value)")
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    # Geographic clustering with K-Means on return metrics
    st.markdown("**Geographic Clustering (K-Means on Return Patterns)**")
    city_cluster_df = df_geo.groupby("City").agg(
        Returns=("OrderID","count"),
        AvgValue=("TotalAmount","mean"),
        AvgQty=("Quantity","mean"),
        AvgPrice=("UnitPrice","mean")).reset_index()

    if len(city_cluster_df) >= 3:
        scaler = StandardScaler()
        X_geo = scaler.fit_transform(city_cluster_df[["Returns","AvgValue","AvgQty"]])
        km = KMeans(n_clusters=min(4, len(city_cluster_df)), random_state=42, n_init=10)
        city_cluster_df["Cluster"] = km.fit_predict(X_geo).astype(str)
        fig4 = px.scatter(city_cluster_df, x="Returns", y="AvgValue", size="AvgQty",
                          color="Cluster", hover_data=["City"],
                          title="City Clusters: Return Volume vs Avg Value",
                          color_discrete_sequence=["#f97316","#6366f1","#22d3ee","#10b981"])
        apply_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**Top Cities Table**")
    st.dataframe(city_agg.reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TEMPORAL TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="tab-header">Temporal Trends</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Time-series analysis · Seasonality · ARIMA-style forecasting</div>', unsafe_allow_html=True)

    # Date range slider
    year_range = st.slider("Year Range", int(df["Year"].min()), int(df["Year"].max()),
                           (int(df["Year"].min()), int(df["Year"].max())), step=1)
    df_t = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

    # Monthly trend per category
    monthly_cat = df_t.groupby(["YearMonth","Category"]).size().reset_index(name="Returns")
    fig = px.line(monthly_cat, x="YearMonth", y="Returns", color="Category",
                  title="Monthly Returns by Category",
                  color_discrete_sequence=["#f97316","#6366f1","#22d3ee","#10b981","#f43f5e","#a78bfa"])
    apply_theme(fig)
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        # Monthly heatmap (year × month)
        hm = df_t.groupby(["Year","Month"]).size().reset_index(name="Returns")
        hm_pivot = hm.pivot(index="Year", columns="Month", values="Returns").fillna(0)
        hm_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"][:len(hm_pivot.columns)]
        fig2 = px.imshow(hm_pivot, text_auto=True, color_continuous_scale="Oranges",
                         title="Seasonal Heatmap (Year × Month)")
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        # Quarterly aggregation
        q_agg = df_t.groupby("Quarter").agg(
            Returns=("OrderID","count"), Revenue=("TotalAmount","sum")).reset_index()
        fig3 = px.bar(q_agg, x="Quarter", y="Returns", color="Revenue",
                      color_continuous_scale="Viridis",
                      title="Returns per Quarter (colored by Revenue)")
        apply_theme(fig3)
        fig3.update_xaxes(tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)

    # Simple forecasting with linear + polynomial trend
    st.markdown("**📈 Trend Forecasting (Linear Regression on Monthly Returns)**")
    monthly_all = df_t.groupby("YearMonth").size().reset_index(name="Returns")
    monthly_all["t"] = np.arange(len(monthly_all))

    if len(monthly_all) > 4:
        from numpy.polynomial import polynomial as P
        t = monthly_all["t"].values
        y = monthly_all["Returns"].values
        # Fit linear
        coeffs1 = np.polyfit(t, y, 1)
        # Fit poly-2
        coeffs2 = np.polyfit(t, y, 2)
        # Forecast 6 months ahead
        t_future = np.arange(len(t), len(t)+6)
        labels_future = [f"F+{i+1}" for i in range(6)]

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=monthly_all["YearMonth"], y=monthly_all["Returns"],
                              name="Actual", marker_color="#f97316", opacity=0.6))
        fig4.add_trace(go.Scatter(x=monthly_all["YearMonth"],
                                  y=np.polyval(coeffs1, t),
                                  name="Linear Trend", line=dict(color="#22d3ee", dash="dash")))
        fig4.add_trace(go.Scatter(x=monthly_all["YearMonth"],
                                  y=np.polyval(coeffs2, t),
                                  name="Poly Trend", line=dict(color="#a78bfa", dash="dot")))
        # Forecast
        forecast_y = np.polyval(coeffs2, t_future)
        forecast_y = np.clip(forecast_y, 0, None)
        fig4.add_trace(go.Scatter(x=labels_future, y=forecast_y,
                                  name="Forecast (6M)", mode="lines+markers",
                                  line=dict(color="#10b981", width=2),
                                  marker=dict(symbol="diamond", size=8)))
        fig4.update_layout(title="Return Volume Forecast (Polynomial Regression)", **PLOTLY_LAYOUT)
        fig4.update_xaxes(gridcolor="#252a35", tickangle=-45)
        fig4.update_yaxes(gridcolor="#252a35")
        st.plotly_chart(fig4, use_container_width=True)

        slope = coeffs1[0]
        direction = "📈 increasing" if slope > 0 else "📉 decreasing"
        st.info(f"**Trend:** Returns are {direction} by ~{abs(slope):.1f} units/month based on linear regression over the selected period.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PRICE & QUANTITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="tab-header">Price & Quantity</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Descriptive stats · Distribution analysis · Price~Quantity regression</div>', unsafe_allow_html=True)

    brand_sel = st.multiselect("Brand Filter", sorted(df["Brand"].unique()),
                               default=sorted(df["Brand"].unique()), key="pq_brand")
    df_pq = df[df["Brand"].isin(brand_sel)]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_pq, x="UnitPrice", nbins=40, color="Category",
                           title="Unit Price Distribution by Category",
                           color_discrete_sequence=["#f97316","#6366f1","#22d3ee","#10b981","#f43f5e","#a78bfa"])
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(df_pq, x="Quantity", nbins=10, color="Category",
                            title="Quantity Distribution by Category",
                            color_discrete_sequence=["#f97316","#6366f1","#22d3ee","#10b981","#f43f5e","#a78bfa"])
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.box(df_pq, x="Brand", y="UnitPrice", color="Brand",
                      title="Unit Price Box Plot by Brand")
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.box(df_pq, x="Category", y="Quantity", color="Category",
                      title="Quantity Box Plot by Category")
        apply_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    # Scatter: UnitPrice vs TotalAmount with regression line
    st.markdown("**Scatter: UnitPrice vs TotalAmount (with Regression)**")
    fig5 = px.scatter(df_pq, x="UnitPrice", y="TotalAmount", color="Category",
                      trendline="ols", opacity=0.6, size="Quantity",
                      title="UnitPrice vs TotalAmount (OLS Trendline per Category)",
                      color_discrete_sequence=["#f97316","#6366f1","#22d3ee","#10b981","#f43f5e","#a78bfa"])
    apply_theme(fig5)
    st.plotly_chart(fig5, use_container_width=True)

    # Correlation summary
    corr = df_pq[["Quantity","UnitPrice","Discount","Tax","ShippingCost","TotalAmount"]].corr()
    col_a, col_b = st.columns(2)
    with col_a:
        fig6 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         title="Price & Quantity Correlation Matrix")
        apply_theme(fig6)
        st.plotly_chart(fig6, use_container_width=True)
    with col_b:
        # Discount impact
        df_pq["DiscountBin"] = pd.cut(df_pq["Discount"], bins=[-0.01,0,0.1,0.2,0.3,1.0],
                                       labels=["0%","1-10%","11-20%","21-30%","30%+"])
        disc_agg = df_pq.groupby("DiscountBin", observed=True).agg(
            Returns=("OrderID","count"), AvgValue=("TotalAmount","mean")).reset_index()
        fig7 = px.bar(disc_agg, x="DiscountBin", y="Returns", color="AvgValue",
                      color_continuous_scale="Oranges",
                      title="Returns by Discount Tier")
        apply_theme(fig7)
        st.plotly_chart(fig7, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ML MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="tab-header">ML Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Linear Regression · Decision Tree · Random Forest · AdaBoost · Gradient Boosting</div>', unsafe_allow_html=True)

    @st.cache_data
    def train_models(data_hash):
        df_ml = df_full.copy()
        le_cat   = LabelEncoder()
        le_brand = LabelEncoder()
        le_pay   = LabelEncoder()
        le_city  = LabelEncoder()
        le_cntry = LabelEncoder()
        df_ml["Cat_enc"]   = le_cat.fit_transform(df_ml["Category"])
        df_ml["Brand_enc"] = le_brand.fit_transform(df_ml["Brand"])
        df_ml["Pay_enc"]   = le_pay.fit_transform(df_ml["PaymentMethod"])
        df_ml["City_enc"]  = le_city.fit_transform(df_ml["City"])
        df_ml["Cntry_enc"] = le_cntry.fit_transform(df_ml["Country"])
        features = ["Cat_enc","Brand_enc","Pay_enc","City_enc","Cntry_enc",
                    "Quantity","UnitPrice","Discount","Tax","ShippingCost",
                    "Year","Month"]
        X = df_ml[features]
        y = df_ml["TotalAmount"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree":     DecisionTreeRegressor(max_depth=8, random_state=42),
            "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "AdaBoost":          AdaBoostRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        results = []
        preds_dict = {}
        importance_dict = {}
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            rmse = np.sqrt(mean_squared_error(y_te, pred))
            r2   = r2_score(y_te, pred)
            results.append({"Model": name, "R²": round(r2,4), "RMSE": round(rmse,2),
                             "Test Size": len(y_te)})
            preds_dict[name] = pred
            if hasattr(model, "feature_importances_"):
                importance_dict[name] = dict(zip(features, model.feature_importances_))
        return pd.DataFrame(results), preds_dict, importance_dict, y_te, X_te, features

    results_df, preds_dict, importance_dict, y_te, X_te, features = train_models(len(df_full))

    # Model comparison table
    best_model = results_df.loc[results_df["R²"].idxmax(), "Model"]
    st.success(f"🏆 Best Model: **{best_model}** — R² = {results_df['R²'].max():.4f}")

    col_tb, col_bar = st.columns([1,2])
    with col_tb:
        st.markdown("**Model Performance Table**")
        st.dataframe(results_df.style.highlight_max(subset=["R²"], color="#1a3a2a")
                                      .highlight_min(subset=["RMSE"], color="#1a3a2a"),
                     use_container_width=True)
    with col_bar:
        fig = px.bar(results_df, x="Model", y="R²", color="RMSE",
                     color_continuous_scale="Oranges_r",
                     title="Model R² Comparison",
                     text=results_df["R²"].apply(lambda x: f"{x:.4f}"))
        fig.update_traces(textposition="outside")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Predicted vs Actual
    st.markdown("**Predicted vs Actual (all models)**")
    fig2 = go.Figure()
    colors = ["#f97316","#6366f1","#22d3ee","#10b981","#f43f5e"]
    for (name, pred), color in zip(preds_dict.items(), colors):
        fig2.add_trace(go.Scatter(x=y_te.values[:200], y=pred[:200],
                                  mode="markers", name=name,
                                  marker=dict(color=color, size=5, opacity=0.6)))
    fig2.add_trace(go.Scatter(x=[y_te.min(), y_te.max()],
                              y=[y_te.min(), y_te.max()],
                              mode="lines", name="Perfect Fit",
                              line=dict(color="white", dash="dash", width=1)))
    fig2.update_layout(title="Predicted vs Actual TotalAmount (first 200 test points)", **PLOTLY_LAYOUT)
    fig2.update_xaxes(title="Actual", gridcolor="#252a35")
    fig2.update_yaxes(title="Predicted", gridcolor="#252a35")
    st.plotly_chart(fig2, use_container_width=True)

    # Feature importance
    st.markdown("**Feature Importance**")
    feat_cols = st.columns(len(importance_dict))
    for i, (mname, imp) in enumerate(importance_dict.items()):
        fi = pd.DataFrame({"Feature": list(imp.keys()), "Importance": list(imp.values())})
        fi = fi.sort_values("Importance", ascending=True)
        fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                      title=mname, color="Importance",
                      color_continuous_scale="Oranges")
        apply_theme(fig3)
        feat_cols[i].plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — RETURN PREDICTOR & SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="tab-header">Return Predictor & Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Random Forest + Gradient Boosting · Interactive prediction · Confidence intervals</div>', unsafe_allow_html=True)

    @st.cache_data
    def build_predictor():
        df_ml = df_full.copy()
        le_cat   = LabelEncoder(); cats   = le_cat.fit_transform(df_ml["Category"])
        le_brand = LabelEncoder(); brands = le_brand.fit_transform(df_ml["Brand"])
        le_pay   = LabelEncoder(); pays   = le_pay.fit_transform(df_ml["PaymentMethod"])
        le_city  = LabelEncoder(); cities = le_city.fit_transform(df_ml["City"])
        le_cntry = LabelEncoder(); cntrs  = le_cntry.fit_transform(df_ml["Country"])
        df_ml["Cat_enc"]   = cats
        df_ml["Brand_enc"] = brands
        df_ml["Pay_enc"]   = pays
        df_ml["City_enc"]  = cities
        df_ml["Cntry_enc"] = cntrs
        features = ["Cat_enc","Brand_enc","Pay_enc","City_enc","Cntry_enc",
                    "Quantity","UnitPrice","Discount","Tax","ShippingCost","Year","Month"]
        X = df_ml[features]; y = df_ml["TotalAmount"]
        rf  = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        gb  = GradientBoostingRegressor(n_estimators=200, random_state=42)
        rf.fit(X, y); gb.fit(X, y)
        return rf, gb, le_cat, le_brand, le_pay, le_city, le_cntry

    rf_pred, gb_pred, le_c, le_b, le_p, le_ci, le_cn = build_predictor()

    st.markdown("### 🎛️ Input Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_cat    = st.selectbox("Category",   sorted(df_full["Category"].unique()))
        sim_brand  = st.selectbox("Brand",       sorted(df_full["Brand"].unique()))
        sim_pay    = st.selectbox("Payment Method", sorted(df_full["PaymentMethod"].unique()))
    with col2:
        sim_city   = st.selectbox("City",        sorted(df_full["City"].unique()))
        sim_cntry  = st.selectbox("Country",     sorted(df_full["Country"].unique()))
        sim_qty    = st.slider("Quantity", 1, 10, 3)
    with col3:
        sim_price  = st.slider("Unit Price ($)", 10.0, 600.0, 150.0, step=10.0)
        sim_disc   = st.slider("Discount",  0.0, 0.5, 0.0, step=0.05)
        sim_tax    = st.slider("Tax ($)",   0.0, 200.0, 30.0, step=5.0)
        sim_ship   = st.slider("Shipping ($)", 0.0, 30.0, 5.0, step=0.5)
        sim_year   = st.selectbox("Year",   [2020,2021,2022,2023,2024])
        sim_month  = st.selectbox("Month",  list(range(1,13)), index=0)

    if st.button("🔮 Predict Return Value", use_container_width=True):
        try:
            x_in = np.array([[
                le_c.transform([sim_cat])[0],
                le_b.transform([sim_brand])[0],
                le_p.transform([sim_pay])[0],
                le_ci.transform([sim_city])[0],
                le_cn.transform([sim_cntry])[0],
                sim_qty, sim_price, sim_disc, sim_tax, sim_ship,
                sim_year, sim_month
            ]])
            rf_out  = rf_pred.predict(x_in)[0]
            gb_out  = gb_pred.predict(x_in)[0]
            ensemble = (rf_out + gb_out) / 2

            # Confidence interval via RF tree variance
            tree_preds = np.array([t.predict(x_in)[0] for t in rf_pred.estimators_])
            ci_low  = np.percentile(tree_preds, 5)
            ci_high = np.percentile(tree_preds, 95)

            st.markdown("<br>", unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Random Forest", f"${rf_out:,.2f}")
            c2.metric("Gradient Boosting", f"${gb_out:,.2f}")
            c3.metric("Ensemble Prediction", f"${ensemble:,.2f}")
            c4.metric("90% CI", f"${ci_low:,.0f} – ${ci_high:,.0f}")

            # Distribution of tree predictions
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=tree_preds, nbinsx=30,
                                       marker_color="#f97316", opacity=0.7, name="Tree Predictions"))
            fig.add_vline(x=ensemble, line_color="#22d3ee", line_width=2,
                          annotation_text=f"Ensemble: ${ensemble:,.0f}", annotation_font_color="#22d3ee")
            fig.add_vline(x=ci_low,  line_color="#10b981", line_dash="dash", line_width=1)
            fig.add_vline(x=ci_high, line_color="#10b981", line_dash="dash", line_width=1)
            fig.update_layout(title="Prediction Distribution (RF Tree Ensemble)", **PLOTLY_LAYOUT)
            fig.update_xaxes(title="Predicted TotalAmount ($)", gridcolor="#252a35")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    # What-If Simulator
    st.divider()
    st.markdown("### 📊 What-If Simulator: Price Sensitivity")
    price_range = np.linspace(10, 600, 60)
    rf_curve, gb_curve = [], []
    try:
        base_row = [
            le_c.transform([sim_cat])[0],
            le_b.transform([sim_brand])[0],
            le_p.transform([sim_pay])[0],
            le_ci.transform([sim_city])[0],
            le_cn.transform([sim_cntry])[0],
            sim_qty, 100, sim_disc, sim_tax, sim_ship, sim_year, sim_month
        ]
        for p in price_range:
            row = base_row.copy(); row[6] = p
            rf_curve.append(rf_pred.predict([row])[0])
            gb_curve.append(gb_pred.predict([row])[0])

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=price_range, y=rf_curve,
                                     name="Random Forest", line=dict(color="#f97316",width=2)))
        fig_sim.add_trace(go.Scatter(x=price_range, y=gb_curve,
                                     name="Gradient Boosting", line=dict(color="#6366f1",width=2)))
        fig_sim.add_trace(go.Scatter(x=price_range, y=[(r+g)/2 for r,g in zip(rf_curve, gb_curve)],
                                     name="Ensemble", line=dict(color="#22d3ee",width=2,dash="dash")))
        fig_sim.update_layout(title="Predicted Return Value vs Unit Price (selected config)",
                              xaxis_title="Unit Price ($)", yaxis_title="Predicted TotalAmount ($)",
                              **PLOTLY_LAYOUT)
        fig_sim.update_xaxes(gridcolor="#252a35")
        fig_sim.update_yaxes(gridcolor="#252a35")
        st.plotly_chart(fig_sim, use_container_width=True)
    except:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — HIGH-RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="tab-header">High-Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">K-Means clustering · Decision Tree classification · Outlier / anomaly detection</div>', unsafe_allow_html=True)

    cat_f8  = st.multiselect("Category Filter",  sorted(df["Category"].unique()),
                              default=sorted(df["Category"].unique()), key="hr_cat")
    brand_f8 = st.multiselect("Brand Filter",    sorted(df["Brand"].unique()),
                               default=sorted(df["Brand"].unique()), key="hr_brand")
    df_hr = df[df["Category"].isin(cat_f8) & df["Brand"].isin(brand_f8)]

    # Top return-prone products
    prod_risk = df_hr.groupby(["ProductName","Category","Brand"]).agg(
        ReturnCount=("OrderID","count"),
        TotalRevenueLost=("TotalAmount","sum"),
        AvgValue=("TotalAmount","mean"),
        AvgQty=("Quantity","mean"),
        AvgPrice=("UnitPrice","mean"),
    ).reset_index().sort_values("ReturnCount", ascending=False)

    col_l, col_r = st.columns([2,1])
    with col_l:
        fig = px.bar(prod_risk.head(20), x="ReturnCount", y="ProductName",
                     orientation="h", color="TotalRevenueLost",
                     color_continuous_scale="Reds",
                     title="Top 20 High-Risk Products by Return Count")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig2 = px.scatter(prod_risk, x="ReturnCount", y="TotalRevenueLost",
                          size="AvgQty", color="Category", hover_data=["ProductName"],
                          title="Risk Matrix: Count vs Revenue Lost",
                          color_discrete_sequence=["#f97316","#6366f1","#22d3ee","#10b981","#f43f5e","#a78bfa"])
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # K-Means clustering on products
    st.markdown("**Product Risk Clusters (K-Means)**")
    if len(prod_risk) >= 4:
        X_hr = StandardScaler().fit_transform(
            prod_risk[["ReturnCount","TotalRevenueLost","AvgPrice"]].fillna(0))
        n_clusters = min(4, len(prod_risk))
        km_hr = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        prod_risk["Cluster"] = km_hr.fit_predict(X_hr).astype(str)
        cluster_labels = {"0":"Low Risk","1":"Medium Risk","2":"High Risk","3":"Critical"}
        prod_risk["RiskLabel"] = prod_risk["Cluster"].map(
            lambda x: cluster_labels.get(x, f"Cluster {x}"))

        fig3 = px.scatter(prod_risk, x="ReturnCount", y="AvgValue",
                          size="TotalRevenueLost", color="RiskLabel",
                          hover_data=["ProductName","Brand"],
                          title="Product Risk Clusters (size = Total Revenue Lost)",
                          color_discrete_map={"Low Risk":"#10b981","Medium Risk":"#fbbf24",
                                              "High Risk":"#f97316","Critical":"#ef4444"})
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    # Anomaly detection via Z-score
    st.markdown("**Anomaly Detection (Z-Score on TotalAmount)**")
    df_hr = df_hr.copy()
    df_hr["Z_Score"] = (df_hr["TotalAmount"] - df_hr["TotalAmount"].mean()) / df_hr["TotalAmount"].std()
    df_hr["Anomaly"] = df_hr["Z_Score"].abs() > 2.5
    anomalies = df_hr[df_hr["Anomaly"]].sort_values("Z_Score", ascending=False)

    col_a, col_b = st.columns(2)
    with col_a:
        fig4 = go.Figure()
        normal = df_hr[~df_hr["Anomaly"]]
        fig4.add_trace(go.Scatter(x=normal["OrderDate"], y=normal["TotalAmount"],
                                  mode="markers", name="Normal",
                                  marker=dict(color="#6366f1", size=4, opacity=0.5)))
        fig4.add_trace(go.Scatter(x=anomalies["OrderDate"], y=anomalies["TotalAmount"],
                                  mode="markers", name="Anomaly",
                                  marker=dict(color="#ef4444", size=8, symbol="x")))
        fig4.update_layout(title="Anomaly Detection (Z-Score > 2.5)", **PLOTLY_LAYOUT)
        fig4.update_xaxes(gridcolor="#252a35")
        fig4.update_yaxes(gridcolor="#252a35")
        st.plotly_chart(fig4, use_container_width=True)
    with col_b:
        fig5 = px.histogram(df_hr, x="Z_Score", nbins=50,
                             color="Anomaly",
                             color_discrete_map={True:"#ef4444", False:"#6366f1"},
                             title="Z-Score Distribution")
        fig5.add_vline(x=2.5,  line_dash="dash", line_color="#fbbf24", annotation_text="+2.5σ")
        fig5.add_vline(x=-2.5, line_dash="dash", line_color="#fbbf24", annotation_text="-2.5σ")
        apply_theme(fig5)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown(f"**{len(anomalies)} anomalous returns detected** (|Z| > 2.5)")
    st.dataframe(anomalies[["OrderID","OrderDate","ProductName","Category","Brand",
                              "TotalAmount","Z_Score"]].head(30).reset_index(drop=True),
                 use_container_width=True)

    st.markdown("**High-Risk Product Table**")
    st.dataframe(prod_risk.head(30).reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — RAW DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div class="tab-header">Raw Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="tab-sub">Filterable dataframe · Download CSV · Column search · Data quality checks</div>', unsafe_allow_html=True)

    # Search
    col_s1, col_s2, col_s3 = st.columns(3)
    search_prod = col_s1.text_input("🔍 Search Product Name", "")
    search_cust = col_s2.text_input("🔍 Search Customer Name", "")
    search_city = col_s3.text_input("🔍 Search City", "")

    df_raw = df.copy()
    if search_prod:
        df_raw = df_raw[df_raw["ProductName"].str.contains(search_prod, case=False, na=False)]
    if search_cust:
        df_raw = df_raw[df_raw["CustomerName"].str.contains(search_cust, case=False, na=False)]
    if search_city:
        df_raw = df_raw[df_raw["City"].str.contains(search_city, case=False, na=False)]

    # Column selector
    all_cols = df_raw.columns.tolist()
    sel_cols = st.multiselect("Select Columns to Display", all_cols, default=all_cols)
    df_display = df_raw[sel_cols] if sel_cols else df_raw

    st.markdown(f"<span class='pill'>{len(df_display):,} rows · {len(sel_cols)} columns</span>", unsafe_allow_html=True)
    st.dataframe(df_display, use_container_width=True, height=400)

    # Download
    csv_bytes = df_display.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered CSV", data=csv_bytes,
                       file_name="amazon_returns_filtered.csv", mime="text/csv",
                       use_container_width=True)

    st.divider()
    st.markdown("**Data Quality Report**")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        null_counts = df_full.isnull().sum().reset_index()
        null_counts.columns = ["Column","Nulls"]
        null_counts["% Missing"] = (null_counts["Nulls"] / len(df_full) * 100).round(2)
        st.dataframe(null_counts, use_container_width=True)
    with col_q2:
        dtype_df = df_full.dtypes.reset_index()
        dtype_df.columns = ["Column","DType"]
        nunique = df_full.nunique().reset_index()
        nunique.columns = ["Column","Unique Values"]
        dq = dtype_df.merge(nunique, on="Column")
        st.dataframe(dq, use_container_width=True)

    # Duplicate check
    dupes = df_full.duplicated().sum()
    st.info(f"🔍 Duplicate rows: **{dupes}** | Total records in dataset: **{len(df_full):,}**")
