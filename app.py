# app.py
import streamlit as st
import pandas as pd
import numpy as np
import glob
import io
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from itertools import combinations
from collections import Counter

st.set_page_config(page_title="Retail Sales Analytics", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Spark initialization
# -----------------------------
MASTER_URL = "spark://10.219.254.67:7077"

try:
    sc = SparkContext(MASTER_URL, "SalesAnalyticsApp")
    spark = SparkSession(sc)
    spark.sparkContext.setLogLevel("ERROR")
    spark_available = True
except Exception as e:
    sc = None
    spark = None
    spark_available = False
    st.sidebar.error(f"Warning: Could not start Spark: {e}")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Sales Analytics Dashboard (2019)")
st.sidebar.markdown("Upload monthly CSV files or let the app auto-detect:")

# Link to your project PDF
pdf_local = "/mnt/data/Lab project work for CSBB 422.pdf"
st.sidebar.markdown(f"[Project Document]({pdf_local})")

uploaded_files = st.sidebar.file_uploader("Upload CSVs (multiple allowed)", type=["csv"], accept_multiple_files=True)

# IMPORTANT: Updated auto-detect pattern for your actual files
use_local_glob = st.sidebar.checkbox("Auto-load local files matching Sales_*_2019*", value=True)

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data(show_spinner=False)
def read_csvs_with_spark(paths):
    if not spark:
        raise RuntimeError("Spark is not available.")
    df_spark = spark.read.option("header", "true").option("inferSchema", "true").csv(paths)
    for c in df_spark.columns:
        df_spark = df_spark.withColumnRenamed(c, c.strip())
    return df_spark.toPandas()

@st.cache_data(show_spinner=False)
def read_uploaded_files_to_pandas(uploaded_files):
    dfs = []
    for f in uploaded_files:
        df = pd.read_csv(f)
        dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def clean_and_engineer(df):
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all")
    # Remove duplicate header rows
    if 'Quantity Ordered' in df.columns:
        df = df[df['Quantity Ordered'].astype(str).str.lower() != 'quantity ordered']

    df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors="coerce").fillna(0).astype(int)
    df['Price Each'] = pd.to_numeric(df['Price Each'], errors="coerce").fillna(0.0)
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors="coerce")

    df['Revenue'] = df['Quantity Ordered'] * df['Price Each']

    # Address parsing
    def parse_city(addr):
        try:
            return addr.split(',')[1].strip()
        except:
            return "Unknown"

    def parse_state(addr):
        try:
            return addr.split(',')[2].strip().split(' ')[0]
        except:
            return "Unknown"

    if "Purchase Address" in df.columns:
        df['City'] = df['Purchase Address'].apply(parse_city)
        df['State'] = df['Purchase Address'].apply(parse_state)
    else:
        df['City'] = "Unknown"
        df['State'] = "Unknown"

    df['Month'] = df['Order Date'].dt.month
    df['Hour'] = df['Order Date'].dt.hour
    df['Day'] = df['Order Date'].dt.day_name()

    return df

# -----------------------------
# Load / Merge Monthly Files
# -----------------------------
st.title("🛒 Retail Sales - 2019 Monthly Analytics Dashboard")

if uploaded_files:
    df_raw = read_uploaded_files_to_pandas(uploaded_files)
elif use_local_glob:
    # 🔥 Updated pattern to match your actual filenames
    # Matches:
    # Sales_January_2019.csv
    # Sales_Februrary_2019.csv
    # Sales_April_2019.csv
    # Sales_Augut_2019.csv
    # Sales_Decemeber_2019.csv
    local_paths = sorted(glob.glob("Sales_*_2019*.csv") + glob.glob("Sales_*_2019*.CSV"))

    if local_paths:
        st.sidebar.success(f"Found {len(local_paths)} files:")
        for p in local_paths:
            st.sidebar.write("📄 " + p)

        if spark_available:
            try:
                df_raw = read_csvs_with_spark(local_paths)
            except Exception as e:
                st.sidebar.error(f"Spark read failed: {e}")
                dfs = [pd.read_csv(p) for p in local_paths]
                df_raw = pd.concat(dfs, ignore_index=True)
        else:
            dfs = [pd.read_csv(p) for p in local_paths]
            df_raw = pd.concat(dfs, ignore_index=True)
    else:
        st.sidebar.warning("No files found matching Sales_*_2019*.csv. Upload manually.")
        df_raw = pd.DataFrame()
else:
    st.sidebar.info("Upload files or enable auto-detect.")
    df_raw = pd.DataFrame()

if df_raw.empty:
    st.warning("No data loaded yet.")
    st.stop()


# show a preview
st.subheader("📌 Raw Merged Preview (first 10 rows)")
st.dataframe(df_raw.head(10))

# -----------------------------
# Missing values summary
# -----------------------------
st.subheader("🚨 Missing Values Check")
mv = df_raw.isnull().sum().to_frame("Missing Count")
colA, colB = st.columns([1,2])
with colA:
    st.dataframe(mv)
with colB:
    st.write("Quick sample of rows with missing critical fields:")
    st.dataframe(df_raw[df_raw[['Order ID','Product','Quantity Ordered','Price Each','Order Date']].isnull().any(axis=1)].head(5))

# -----------------------------
# Cleaning & feature engineering
# -----------------------------
df = clean_and_engineer(df_raw)

# Validate required columns
required = ['Order ID','Product','Quantity Ordered','Price Each','Order Date','Revenue']
missing_required = [c for c in required if c not in df.columns]
if missing_required:
    st.error(f"Missing required columns after read: {missing_required}. Please check your CSV columns.")
    st.stop()

# -----------------------------
# KPIs
# -----------------------------
total_revenue = df['Revenue'].sum()
total_orders = df['Order ID'].nunique()
total_items = df['Quantity Ordered'].sum()
aov = total_revenue / max(total_orders, 1)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", f"${total_revenue:,.2f}")
k2.metric("Total Orders", f"{total_orders:,}")
k3.metric("Total Items Sold", f"{total_items:,}")
k4.metric("Avg Order Value (AOV)", f"${aov:,.2f}")

st.markdown("---")

# -----------------------------
# Visualizations
# -----------------------------
left, right = st.columns((2,1))

with left:
    st.subheader("📈 Monthly Revenue (Bar)")
    monthly = df.groupby("Month", sort=True)['Revenue'].sum().reset_index().sort_values("Month")
    # If months are sparse (user said january, feb, april, august, december) we still plot them
    fig_month = px.bar(monthly, x='Month', y='Revenue',
                       labels={'Month':'Month (1=Jan)', 'Revenue':'Revenue ($)'},
                       text=monthly['Revenue'].map(lambda x: f"${x:,.0f}"),
                       title="Revenue by Month")
    st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("🔥 Top Products by Units Sold (Bar) & Revenue Share (Pie)")
    prod_stats = df.groupby("Product").agg(
        units_sold=('Quantity Ordered','sum'),
        revenue=('Revenue','sum')
    ).reset_index().sort_values('units_sold', ascending=False)

    top_n = st.slider("Top N products to show", 5, 20, 10)
    top_products = prod_stats.head(top_n)

    fig_prod_bar = px.bar(top_products, x='Product', y='units_sold', text='units_sold',
                          title=f"Top {top_n} Products by Units Sold")
    fig_prod_bar.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig_prod_bar, use_container_width=True)

    # Pie chart for revenue share (top 8 + other)
    st.markdown("**Revenue Distribution (Top products)**")
    top_rev = prod_stats.sort_values('revenue', ascending=False).head(8)
    others = prod_stats['revenue'].sum() - top_rev['revenue'].sum()
    pie_df = pd.concat([top_rev, pd.DataFrame([{'Product':'Other','units_sold':0,'revenue':others}])], ignore_index=True)
    fig_pie = px.pie(pie_df, values='revenue', names='Product', title="Revenue Share (Top 8 + Other)", hole=0.35)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("🕑 Orders by Hour (Line)")
    hourly = df.groupby("Hour")['Order ID'].count().reset_index().sort_values("Hour")
    fig_hour = px.line(hourly, x='Hour', y='Order ID', markers=True, title="Orders by Hour of Day")
    fig_hour.update_yaxes(title_text="Number of Orders")
    st.plotly_chart(fig_hour, use_container_width=True)

with right:
    st.subheader("📍 Revenue by City (Bar)")
    city_rev = df.groupby("City")['Revenue'].sum().reset_index().sort_values('Revenue', ascending=False)
    fig_city = px.bar(city_rev.head(15), x='City', y='Revenue', title="Top Cities by Revenue",
                      text=city_rev['Revenue'].map(lambda x: f"${x:,.0f}"))
    st.plotly_chart(fig_city, use_container_width=True)

    st.subheader("🔎 Quick Table: Top Products (revenue & units)")
    st.dataframe(prod_stats.head(20))

# -----------------------------
# Frequently bought together (basic)
# -----------------------------
st.markdown("---")
st.subheader("🧩 Frequently Bought Together (Pairs)")

# Find order IDs that have multiple items -> group and count combos
df_dup = df[df['Order ID'].duplicated(keep=False)].copy()
if not df_dup.empty:
    df_dup['Grouped'] = df_dup.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
    df_bundles = df_dup[['Order ID','Grouped']].drop_duplicates()
    pair_counter = Counter()
    for row in df_bundles['Grouped']:
        items = row.split(',')
        # count unique pairs (order-insensitive)
        pair_counter.update(Counter(combinations(items, 2)))
    top_pairs = pair_counter.most_common(10)
    if top_pairs:
        pair_df = pd.DataFrame([{'pair': f"{a} + {b}", 'count': c} for (a,b),c in top_pairs])
        st.table(pair_df)
    else:
        st.info("No product pairs found.")
else:
    st.info("No orders with multiple products found; cannot compute bundles.")

# -----------------------------
# Correlation heatmap (numeric features)
# -----------------------------
st.markdown("---")
st.subheader("🌡 Correlation Heatmap (Numeric Features)")

numeric_cols = ['Quantity Ordered', 'Price Each', 'Revenue', 'Month', 'Hour']
heat_df = df[numeric_cols].copy().dropna()
if heat_df.shape[0] > 1:
    corr = heat_df.corr()
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="vlag", ax=ax)
    st.pyplot(fig)
else:
    st.info("Not enough numeric data for correlation heatmap.")

# -----------------------------
# Export aggregated results option
# -----------------------------
st.markdown("---")
st.subheader("💾 Export aggregated CSVs")
if st.button("Save top products & monthly aggregates to CSV"):
    out_prod = prod_stats.sort_values('revenue', ascending=False)
    out_month = monthly.sort_values('Month')
    out_prod.to_csv("agg_top_products.csv", index=False)
    out_month.to_csv("agg_monthly_revenue.csv", index=False)
    st.success("Saved agg_top_products.csv and agg_monthly_revenue.csv to app folder.")

st.success("✔ Dashboard loaded successfully!")

# Note: Do not stop SparkContext here; Streamlit app is live. If you need to stop, call sc.stop() in a teardown.
