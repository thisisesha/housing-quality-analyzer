# 🏡 Miami Housing Quality Analyzer

An interactive Streamlit dashboard to explore real estate trends in Miami, FL. This tool analyzes housing quality through multiple lenses: neighborhood desirability, market value factors, property age, and seasonality in home sales.

## 📊 Features

- **🏘️ Neighborhood Quality Index (NQI)**  
  Combines property size, price, and proximity to amenities (ocean, city center, highways, etc.) into a composite regional score.

- **📈 Effects on Market Value**  
  Analyzes how living area, land size, and structure quality impact sale prices using correlation heatmaps and regression.

- **🏚️ Age-Related Depreciation Trends**  
  Groups homes by age and investigates pricing patterns and depreciation trends across categories.

- **📅 Seasonality in Sales**  
  Identifies how sale prices vary by month using statistical analysis and seasonal clustering (KMeans).

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/thisisesha/housing-quality-analyzer.git
cd miami-housing-quality-analyzer
```

### 2. Install Requirements
Make sure you have Python 3.8+ installed. Then install the required packages:

```bash
pip install streamlit pandas matplotlib seaborn scikit-learn plotly pydeck altair
```

### 3. Add Dataset
Place the miami-housing.csv dataset in the root directory of the project.

### 4. Run the App
```bash
streamlit run app.py
```

Then open your browser and go to:
http://localhost:8501
