import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import pydeck as pdk
import altair as alt
import plotly.express as px
from sklearn.metrics import r2_score
from scipy.stats import f_oneway

# --- Page config ---
st.set_page_config(page_title="Miami Housing Quality", layout="wide")

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("miami-housing.csv")  # Replace with your actual CSV file name
    return df

df = load_data()

# --- Sidebar navigation ---
st.sidebar.title("🏡 Housing Quality App")
option = st.sidebar.radio("Select a feature:", [
    "Neighborhood Quality Index",
    "Effects on Market Value",
    "Age-Related Depreciation Trends",
    "Seasonality in Property Sales"
])

# --- Feature 1: Neighborhood Quality Index ---
if option == "Neighborhood Quality Index":
    st.header("📊 Neighborhood Quality Index (NQI)")
    st.write("This feature visualizes a composite quality score per ZIP code based on housing features.")
    
    df_nqi = df.copy()

    # Drop rows with missing relevant fields
    df_nqi = df_nqi.dropna(subset=[
        "SALE_PRC", "LND_SQFOOT", "OCEAN_DIST", "RAIL_DIST", "CNTR_DIST", "HWY_DIST"
    ])

    # Normalize features
    def normalize(col):
        return (col - col.min()) / (col.max() - col.min())

    def inverse_normalize(col):  # smaller distance is better
        return 1 - normalize(col)

    df_nqi["price_inv_norm"] = inverse_normalize(df_nqi["SALE_PRC"])
    df_nqi["land_norm"] = normalize(df_nqi["LND_SQFOOT"])
    df_nqi["ocean_inv_norm"] = inverse_normalize(df_nqi["OCEAN_DIST"])
    df_nqi["rail_inv_norm"] = inverse_normalize(df_nqi["RAIL_DIST"])
    df_nqi["center_inv_norm"] = inverse_normalize(df_nqi["CNTR_DIST"])
    df_nqi["highway_inv_norm"] = inverse_normalize(df_nqi["HWY_DIST"])

    # Calculate NQI score
    df_nqi["NQI"] = (
        0.25 * df_nqi["land_norm"] +
        0.25 * df_nqi["price_inv_norm"] +
        0.15 * df_nqi["ocean_inv_norm"] +
        0.1  * df_nqi["rail_inv_norm"] +
        0.15 * df_nqi["center_inv_norm"] +
        0.1  * df_nqi["highway_inv_norm"]
    )

    # Cluster properties based on coordinates
    coords = df_nqi[["LATITUDE", "LONGITUDE"]]
    kmeans = KMeans(n_clusters=10, random_state=42)
    df_nqi["RegionCluster"] = kmeans.fit_predict(coords)

    # Compute average NQI per cluster
    region_nqi = df_nqi.groupby("RegionCluster")["NQI"].mean().sort_values(ascending=False).reset_index()
    region_nqi["NQI"] = region_nqi["NQI"].round(4)

    # --- UI Enhancements ---

    st.subheader("Top 10 Regional Clusters by Neighborhood Quality Index")

    # Expand DataFrame display width
    st.markdown("""
        <style>
        .dataframe-container {
            width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Show wider top-10 table
    # Add labeled index as "Sr. No" without adding a new column
    region_nqi_display = region_nqi.head(10).reset_index(drop=True)
    region_nqi_display.index += 1  # Start from 1 instead of 0
    region_nqi_display.index.name = "Sr. No"

    st.dataframe(region_nqi_display, use_container_width=True)

    # Custom bar chart using Altair
    st.subheader("📈 Bar Chart of Top 10 Regional Clusters by NQI")

    top10_chart = alt.Chart(region_nqi.head(10)).mark_bar(color='#3182bd').encode(
        x=alt.X("RegionCluster:N", title="Region Cluster"),
        y=alt.Y("NQI:Q", title="Average Neighborhood Quality Index"),
        tooltip=["RegionCluster", "NQI"]
    ).properties(
        width=600,
        height=400
    )

    text_labels = alt.Chart(region_nqi.head(10)).mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(
        x=alt.X("RegionCluster:N"),
        y=alt.Y("NQI:Q"),
        text=alt.Text("NQI:Q")
    )

    st.altair_chart(top10_chart + text_labels, use_container_width=True)

    # --- Map Visualization ---

    st.subheader("🗺️ Visualizing Regional Clusters on the Map")

    cluster_centroids = df_nqi.groupby("RegionCluster")[["LATITUDE", "LONGITUDE"]].mean().reset_index()
    cluster_nqi = df_nqi.groupby("RegionCluster")["NQI"].mean().reset_index()
    cluster_centroids = cluster_centroids.merge(cluster_nqi, on="RegionCluster")
    cluster_centroids["NQI"] = cluster_centroids["NQI"].round(4)

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=cluster_centroids["LATITUDE"].mean(),
            longitude=cluster_centroids["LONGITUDE"].mean(),
            zoom=10,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=cluster_centroids,
                get_position='[LONGITUDE, LATITUDE]',
                get_fill_color='[200, 30, 0, 160]',
                get_radius=800,
                pickable=True,
            ),
        ],
        tooltip={"text": "Cluster {RegionCluster}\nAvg NQI: {NQI}"}
    ))


# --- Feature 2: Effects on Market Value ---
elif option == "Effects on Market Value":
    st.header("📈 Effects of Property Characteristics on Market Value")

    df_value = df.copy()
    st.write("We analyze how living area, land size, and structure quality impact sale price.")

    # Drop rows with missing values in relevant columns
    df_value = df_value.dropna(subset=["SALE_PRC", "TOT_LVG_AREA", "LND_SQFOOT", "structure_quality"])

    # --- Correlation Matrix ---
    st.subheader("🔗 Correlation Analysis")
    corr_cols = ["SALE_PRC", "TOT_LVG_AREA", "LND_SQFOOT", "structure_quality"]
    corr_matrix = df_value[corr_cols].corr()

    st.dataframe(corr_matrix.style.background_gradient(cmap="Blues"), use_container_width=True)

    # --- Scatter Plots ---
    st.subheader("📊 Scatterplots")
    scatter_cols = {
        "Total Living Area": "TOT_LVG_AREA",
        "Land Square Footage": "LND_SQFOOT",
        "Structure Quality": "structure_quality"
    }

    for label, col in scatter_cols.items():
        fig = px.scatter(
            df_value, x=col, y="SALE_PRC", trendline="ols",
            title=f"{label} vs Sale Price",
            labels={col: label, "SALE_PRC": "Sale Price ($)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Multiple Linear Regression ---
    st.subheader("📉 Multiple Linear Regression")
    X = df_value[["TOT_LVG_AREA", "LND_SQFOOT", "structure_quality"]]
    y = df_value["SALE_PRC"]

    model = LinearRegression()
    model.fit(X, y)

    # Create readable feature labels
    feature_labels = {
        "TOT_LVG_AREA": "Total Living Area",
        "LND_SQFOOT": "Land Square Footage",
        "structure_quality": "Structure Quality"
    }

    coef_df = pd.DataFrame({
        "Feature": [feature_labels[col] for col in X.columns],
        "Coefficient": model.coef_,
        "Interpretation": [
            "Price increases by ${:,.2f} per sq ft of living area".format(model.coef_[0]),
            "Price increases by ${:,.2f} per sq ft of land".format(model.coef_[1]),
            "Price increases by ${:,.2f} per quality point".format(model.coef_[2])
        ]
    })

    st.dataframe(coef_df, use_container_width=True)

    r2 = model.score(X, y)
    st.write(f"**R² Score:** {r2:.3f} — this indicates that the model explains about {r2*100:.1f}% of the variance in sale prices.")
    
    

# --- Feature 3: Age-Related Depreciation Trends ---
elif option == "Age-Related Depreciation Trends":
    st.header("🏚️ Age-Related Depreciation Trends in Property Pricing")

    df_age = df.copy()
    df_age = df_age.dropna(subset=["age", "SALE_PRC"])

    # --- Age Bucketing ---
    def categorize_age(age):
        if age <= 5:
            return "New (0–5)"
        elif age <= 20:
            return "Moderate (6–20)"
        elif age <= 50:
            return "Old (21–50)"
        else:
            return "Vintage (51+)"
    
    df_age["age_bucket"] = df_age["age"].apply(categorize_age)

    # --- Summary statistics by age bucket ---
    st.subheader("📊 Summary Stats by Age Bucket")
    age_summary = df_age.groupby("age_bucket")["SALE_PRC"].agg(["mean", "median", "std"]).reset_index()
    age_summary.columns = ["Age Bucket", "Mean Price", "Median Price", "Std Dev"]
    st.dataframe(age_summary, use_container_width=True)

    # --- Scatter Plot: Age vs Sale Price ---
    st.subheader("🔍 Scatter Plot: Age vs Sale Price")
    fig1 = px.scatter(
        df_age, x="age", y="SALE_PRC", opacity=0.5,
        trendline="ols", labels={"SALE_PRC": "Sale Price", "age": "Property Age"},
        title="Scatter Plot with Linear Regression"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Boxplot: Age Bucket vs Sale Price ---
    st.subheader("📦 Boxplot: Age Bucket vs Sale Price")
    fig2 = px.box(
        df_age, x="age_bucket", y="SALE_PRC",
        labels={"SALE_PRC": "Sale Price", "age_bucket": "Age Category"},
        title="Boxplot of Sale Price by Age Category"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Line Plot: Mean Price by Age Bucket ---
    st.subheader("📈 Mean Sale Price by Age Bucket")
    age_mean = df_age.groupby("age_bucket")["SALE_PRC"].mean().reset_index()
    fig3 = px.line(
        age_mean, x="age_bucket", y="SALE_PRC",
        markers=True,
        labels={"age_bucket": "Age Category", "SALE_PRC": "Mean Sale Price"},
        title="Line Graph of Mean Price by Age Group"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # --- Regression Model: Age vs Sale Price ---
    st.subheader("📉 Linear Regression: Predicting Price from Age")
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    X = df_age[["age"]]
    y = df_age["SALE_PRC"]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    st.write(f"**Regression Coefficient:** ${model.coef_[0]:,.2f} per year of age")
    st.write(f"**Intercept:** ${model.intercept_:,.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

# --- Feature 4: Seasonality in Property Sales ---
elif option == "Seasonality in Property Sales":
    st.header("📅 Seasonality in Property Sales")

    df_season = df.copy()
    df_season = df_season.dropna(subset=["month_sold", "SALE_PRC"])

    # --- Convert to integer month if needed ---
    df_season["month_sold"] = df_season["month_sold"].astype(int)

    # --- Month labels ---
    month_labels = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    df_season["month_label"] = df_season["month_sold"].map(month_labels)

    # --- Summary statistics ---
    st.subheader("📊 Sale Price Summary by Month")
    month_stats = df_season.groupby("month_label")["SALE_PRC"].agg(["mean", "median", "std"]).reset_index()
    month_stats.columns = ["Month", "Mean Price", "Median Price", "Std Dev"]
    month_stats["Month"] = pd.Categorical(month_stats["Month"], categories=month_labels.values(), ordered=True)
    month_stats = month_stats.sort_values("Month")
    st.dataframe(month_stats, use_container_width=True)

    # --- Boxplot: Monthly Sale Prices ---
    st.subheader("📦 Boxplot of Sale Prices by Month")
    fig_box = px.box(
        df_season, x="month_label", y="SALE_PRC",
        labels={"month_label": "Month", "SALE_PRC": "Sale Price"},
        title="Distribution of Sale Prices by Month"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # --- Line plot: Mean Sale Price by Month ---
    st.subheader("📈 Average Sale Price by Month")
    mean_prices = df_season.groupby("month_label")["SALE_PRC"].mean().reset_index()
    mean_prices["month_label"] = pd.Categorical(mean_prices["month_label"], categories=month_labels.values(), ordered=True)
    mean_prices = mean_prices.sort_values("month_label")

    fig_line = px.line(
        mean_prices, x="month_label", y="SALE_PRC", markers=True,
        labels={"month_label": "Month", "SALE_PRC": "Mean Sale Price"},
        title="Mean Sale Price Trend Over the Year"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # --- Hypothesis Test: ANOVA ---
    st.subheader("🧪 Statistical Test: Do Prices Differ by Month?")
    from scipy.stats import f_oneway

    # Collect price lists for each month
    price_lists = [df_season[df_season["month_sold"] == m]["SALE_PRC"] for m in range(1, 13)]

    # Run ANOVA
    f_stat, p_value = f_oneway(*price_lists)
    st.write(f"**F-Statistic:** {f_stat:.2f}")
    st.write(f"**p-value:** {p_value:.4f}")

    if p_value < 0.05:
        st.success("Result: Sale prices **do vary significantly** across months.")
    else:
        st.info("Result: No significant difference in sale prices between months.")
