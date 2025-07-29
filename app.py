import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import pydeck as pdk
import altair as alt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import f_oneway

# --- Page config setting in Streamlit ---
st.set_page_config(page_title="Miami Housing Quality", layout="wide")

# --- Loading dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("miami-housing.csv")  
    return df

df = load_data()

# --- Sidebar navigation code ---
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
    st.write("This feature combines property size, price, and proximity to key amenities into a single Neighborhood Quality Index, and also explores how distance to amenities affects sale price.")

    # Data cleaning - Making a working copy and dropping missing values.
    df_nqi = df.copy()
    df_nqi = df_nqi.dropna(subset=[
        "SALE_PRC", "LND_SQFOOT", "OCEAN_DIST", "RAIL_DIST", "CNTR_DIST", "HWY_DIST"
    ])

    # --- Part A: Impact of Proximity to Amenities on Sale Price ---
    st.subheader("🔍 Impact of Proximity to Amenities on Sale Price")
    
    # 1. Descriptive statistics
    stats = {
        "Ocean Distance": df_nqi["OCEAN_DIST"],
        "Highway Distance": df_nqi["HWY_DIST"],
        "City Center Distance": df_nqi["CNTR_DIST"],
        "Sale Price": df_nqi["SALE_PRC"]
    }
    desc = []
    for name, col in stats.items():
        desc.append({
            "Feature": name,
            "Mean": f"{col.mean():,.2f}",
            "Median": f"{col.median():,.2f}",
            "Std Dev": f"{col.std():,.2f}"
        })

    st.write("**Units:** Sale Price in USD ($), Distances in meters, Land Area in ft²")
    st.table(pd.DataFrame(desc))

    # 2. Correlations
    correlations = {
        "Ocean Distance": df_nqi["OCEAN_DIST"].corr(df_nqi["SALE_PRC"]),
        "Highway Distance": df_nqi["HWY_DIST"].corr(df_nqi["SALE_PRC"]),
        "City Center Distance": df_nqi["CNTR_DIST"].corr(df_nqi["SALE_PRC"])
    }
    corr_df = pd.DataFrame([
        {"Feature": f, "Correlation with Price": round(c, 3)}
        for f, c in correlations.items()
    ])
    st.table(corr_df)

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    corr_matrix = df_nqi[["SALE_PRC", "OCEAN_DIST", "HWY_DIST", "CNTR_DIST"]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation: Sale Price vs. Distances")
    st.pyplot(fig)

    # 4. Scatter plots
    st.subheader("📈 Scatter Plots: Distance vs. Sale Price")
    for col, label in [("OCEAN_DIST", "Ocean Distance"),
                       ("HWY_DIST", "Highway Distance"),
                       ("CNTR_DIST", "City Center Distance")]:
        fig_scatter = px.scatter(
            df_nqi, x=col, y="SALE_PRC", trendline="ols",
            trendline_color_override="red",
            labels={col: label, "SALE_PRC": "Sale Price ($)"},
            title=f"{label} vs. Sale Price"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Part B: Computing and Visualizing NQI ---

    # Normalizing features
    def normalize(col): return (col - col.min()) / (col.max() - col.min())
    def inv_norm(col): return 1 - normalize(col)

    df_nqi["price_inv_norm"]   = inv_norm(df_nqi["SALE_PRC"])
    df_nqi["land_norm"]        = normalize(df_nqi["LND_SQFOOT"])
    df_nqi["ocean_inv_norm"]   = inv_norm(df_nqi["OCEAN_DIST"])
    df_nqi["rail_inv_norm"]    = inv_norm(df_nqi["RAIL_DIST"])
    df_nqi["center_inv_norm"]  = inv_norm(df_nqi["CNTR_DIST"])
    df_nqi["highway_inv_norm"] = inv_norm(df_nqi["HWY_DIST"])

    df_nqi["NQI"] = (
        0.25 * df_nqi["land_norm"] +
        0.25 * df_nqi["price_inv_norm"] +
        0.15 * df_nqi["ocean_inv_norm"] +
        0.10 * df_nqi["rail_inv_norm"] +
        0.15 * df_nqi["center_inv_norm"] +
        0.10 * df_nqi["highway_inv_norm"]
    )

    # Clustering by location
    coords = df_nqi[["LATITUDE", "LONGITUDE"]]
    kmeans = KMeans(n_clusters=10, random_state=42)
    df_nqi["RegionCluster"] = kmeans.fit_predict(coords)

    region_nqi = (
        df_nqi.groupby("RegionCluster")["NQI"]
        .mean().round(4)
        .sort_values(ascending=False)
        .reset_index()
    )

    # Table with Sr. No index
    region_nqi_display = region_nqi.head(10).reset_index(drop=True)
    region_nqi_display.index += 1
    region_nqi_display.index.name = "Sr. No"

    st.subheader("📊 Top 10 Regional Clusters by NQI")
    st.markdown(
        "<style>.dataframe-container {width:100% !important;}</style>",
        unsafe_allow_html=True
    )
    st.dataframe(region_nqi_display, use_container_width=True)

    # Altair bar chart
    st.subheader("📈 Bar Chart of Top 10 Clusters by NQI")
    chart_data = region_nqi.head(10).assign(ClusterID=lambda d: d.RegionCluster + 1)
    bar = (
        alt.Chart(chart_data)
        .mark_bar(color="#3182bd")
        .encode(
            x=alt.X("ClusterID:O", title="Region Cluster (1–10)"),
            y=alt.Y("NQI:Q", title="Avg. NQI"),
            tooltip=["ClusterID", "NQI"]
        )
    )
    labels = bar.mark_text(dy=-5).encode(text="NQI:Q")
    st.altair_chart(bar + labels, use_container_width=True)

    # Map visualization
    cluster_centroids = (
        df_nqi.groupby("RegionCluster")[["LATITUDE", "LONGITUDE"]]
        .mean().reset_index()
        .merge(region_nqi, on="RegionCluster")
    )
    cluster_centroids["NQI"] = cluster_centroids["NQI"].round(4)

    st.subheader("🗺️ Regional Clusters Map")
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=cluster_centroids.LATITUDE.mean(),
                longitude=cluster_centroids.LONGITUDE.mean(),
                zoom=10
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=cluster_centroids,
                    get_position="[LONGITUDE, LATITUDE]",
                    get_fill_color="[200, 30, 0, 160]",
                    get_radius=800,
                    pickable=True
                ),
            ],
            tooltip={"text": "Cluster {RegionCluster}\nAvg NQI: {NQI}"}
        )
    )


# --- Feature 2: Effects on Market Value ---
elif option == "Effects on Market Value":
    st.header("📈 Effects of Property Characteristics on Market Value")
    st.write(
        "In this section, we explore how three key features—**total living area**, **land size**, "
        "and **structure quality**—correlate with sale price, and build a regression model to "
        "quantify their individual impacts."
    )

    # 1. Preparing and cleaning data
    df_value = df.dropna(subset=["SALE_PRC", "TOT_LVG_AREA", "LND_SQFOOT", "structure_quality"])
    X = df_value[["TOT_LVG_AREA", "LND_SQFOOT", "structure_quality"]]
    y = df_value["SALE_PRC"]

    # 2. Correlation heatmap
    st.subheader("🔗 Correlation Heatmap")
    corr_matrix = df_value[["SALE_PRC", "TOT_LVG_AREA", "LND_SQFOOT", "structure_quality"]].corr()
    fig_corr, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig_corr)

    # 3. Scatterplots
    st.subheader("📊 Scatterplots with Regression Lines")
    scatter_cols = {
        "Total Living Area": "TOT_LVG_AREA",
        "Land Square Footage": "LND_SQFOOT",
        "Structure Quality": "structure_quality"
    }
    for label, col in scatter_cols.items():
        fig_scatter = px.scatter(
            df_value,
            x=col, y="SALE_PRC",
            trendline="ols",
            trendline_color_override="red",
            labels={col: label, "SALE_PRC": "Sale Price ($)"},
            title=f"{label} vs Sale Price"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 4. Fit regression model
    st.subheader("📉 Multiple Linear Regression Results")
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # 5. Coefficients table
    feature_labels = {
        "TOT_LVG_AREA": "Total Living Area",
        "LND_SQFOOT": "Land Square Footage",
        "structure_quality": "Structure Quality"
    }
    coef_df = pd.DataFrame({
        "Feature": [feature_labels[c] for c in X.columns],
        "Coefficient": model.coef_,
        "Interpretation": [
            f"Price increases by ${model.coef_[0]:,.2f} per sq ft living area",
            f"Price increases by ${model.coef_[1]:,.2f} per sq ft land",
            f"Price increases by ${model.coef_[2]:,.2f} per quality point"
        ]
    })
    st.dataframe(coef_df, use_container_width=True)

    # 6. Intercept, MSE, R²
    intercept = model.intercept_
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    st.write(f"**Intercept:** ${intercept:,.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:,.2f}")
    st.write(f"**R² Score:** {r2:.3f}  (Explains {r2*100:.1f}% of variance)")

    # 7. Actual vs Predicted plot
    st.subheader("🔍 Actual vs Predicted Sale Prices")
    fig_ap = px.scatter(
        x=y, y=y_pred,
        labels={"x": "Actual Sale Price ($)", "y": "Predicted Sale Price ($)"},
        title="Actual vs Predicted Sale Prices"
    )
    # adding perfect-fit line
    fig_ap.add_shape(
        type="line",
        x0=y.min(), y0=y.min(),
        x1=y.max(), y1=y.max(),
        line=dict(color="green", dash="dash")
    )
    fig_ap.update_traces(marker=dict(color="blue", opacity=0.6))
    st.plotly_chart(fig_ap, use_container_width=True)
    

# --- Feature 3: Age-Related Depreciation Trends ---
elif option == "Age-Related Depreciation Trends":
    st.header("🏚️ Age-Related Depreciation Trends in Property Pricing")
    st.write(
        "In this section, we investigate how a property's age correlates with its sale price. "
        "By grouping homes into age buckets (New, Moderate, Old, Vintage), visualizing their "
        "price distributions, and fitting a regression line, we can quantify depreciation "
        "or appreciation trends over time."
    )

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

    # --- Summarising statistics by age bucket ---
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
    st.write(
        "This section explores how sale prices fluctuate throughout the year. "
        "We summarize monthly sale price statistics, visualize distributions and trends, "
        "test for significant differences, and cluster months into seasonal groups."
    )

    df_season = df.copy()
    df_season = df_season.dropna(subset=["month_sold", "SALE_PRC"])
    df_season["month_sold"] = df_season["month_sold"].astype(int)

    # Month name mapping
    month_labels = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    df_season["month_label"] = df_season["month_sold"].map(month_labels)

    # Grouping by month
    month_stats = df_season.groupby("month_sold")["SALE_PRC"].agg(["mean", "median", "std"]).reset_index()
    month_stats.columns = ["month_sold", "Mean", "Median", "Std"]
    month_stats["Month"] = month_stats["month_sold"].map(month_labels)

    # --- KMeans Clustering on monthly sale stats ---
    kmeans = KMeans(n_clusters=3, random_state=42)
    month_stats["SeasonCluster"] = kmeans.fit_predict(month_stats[["Mean", "Median", "Std"]])

    # Attach cluster info back to main DataFrame
    df_season = df_season.merge(month_stats[["month_sold", "SeasonCluster"]], on="month_sold", how="left")

    # --- Display cluster-labeled table ---
    st.subheader("📊 Monthly Sale Stats with Seasonal Clustering")
    month_stats_display = month_stats.copy()
    month_stats_display["Month"] = pd.Categorical(month_stats_display["Month"], categories=month_labels.values(), ordered=True)
    month_stats_display = month_stats_display.sort_values("Month")
    st.dataframe(month_stats_display[["Month", "Mean", "Median", "Std", "SeasonCluster"]], use_container_width=True)

    # --- Boxplot with cluster coloring ---
    st.subheader("📦 Sale Price Distribution by Month")
    fig_box = px.box(
        df_season,
        x="month_label", y="SALE_PRC", color="SeasonCluster",
        labels={"month_label": "Month", "SALE_PRC": "Sale Price"},
        title="Sale Price Distribution by Month with Seasonal Clustering"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # --- Line plot: Mean Sale Price by Month ---
    st.subheader("📈 Mean Sale Price Trend by Month")
    mean_prices = df_season.groupby(["month_label", "SeasonCluster"])["SALE_PRC"].mean().reset_index()
    mean_prices["month_label"] = pd.Categorical(mean_prices["month_label"], categories=month_labels.values(), ordered=True)
    mean_prices = mean_prices.sort_values("month_label")

    fig_line = px.line(
        mean_prices, x="month_label", y="SALE_PRC", color="SeasonCluster", markers=True,
        labels={"month_label": "Month", "SALE_PRC": "Mean Sale Price"},
        title="Mean Sale Price Trend by Clustered Months"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # --- Hypothesis Test: ANOVA ---
    st.subheader("🧪 Statistical Test: Do Sale Prices Differ by Month?")
    
    price_lists = [df_season[df_season["month_sold"] == m]["SALE_PRC"] for m in range(1, 13)]
    f_stat, p_value = f_oneway(*price_lists)

    st.write(f"**F-Statistic:** {f_stat:.2f}")
    st.write(f"**p-value:** {p_value:.4f}")

    if p_value < 0.05:
        st.success("Result: Sale prices **do vary significantly** across months.")
    else:
        st.info("Result: No significant difference in sale prices between months.")
