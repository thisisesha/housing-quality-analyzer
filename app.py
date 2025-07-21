import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import pydeck as pdk
import altair as alt
import plotly.express as px

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
    "Feature Correlation Heatmap",
    "Price Distribution by ZIP Code"
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
    
    

# --- Feature 3: Feature Correlation Heatmap ---
elif option == "Feature Correlation Heatmap":
    st.header("🔗 Feature Correlation Heatmap")
    st.write("Understand how housing features relate to each other.")
    # TO DO: Add correlation matrix and heatmap

# --- Feature 4: Price Distribution by ZIP Code ---
elif option == "Price Distribution by ZIP Code":
    st.header("🏷️ Price Distribution by ZIP Code")
    st.write("Select ZIP codes to see the price distribution across properties.")
    # TO DO: Add ZIP code selection and histogram
