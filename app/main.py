# app/main.py

from scipy import stats
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to sys.path to allow importing 'app' as a package
# This is necessary when running streamlit from the project root (e.g., streamlit run app/main.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions from utils.py
from app import utils

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Solar Data Insights Dashboard")

# --- Title and Introduction ---
st.title("☀️ Solar Data Insights Dashboard")
st.markdown("""
Welcome to the interactive dashboard for exploring solar irradiance and environmental data.
Use the sidebar to select a country and customize the displayed data.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Dashboard Controls")

# Country selection
selected_country = st.sidebar.selectbox(
    "Select a Country:",
    utils.get_country_list()
)

# Load data for the selected country
# Using st.cache_data to cache the DataFrame for performance
@st.cache_data
def get_data(country: str):
    """Loads and cleans data for the selected country."""
    df = utils.load_data(country)
    # Perform cleaning process on the loaded (simulated) data
    df_cleaned = utils.perform_cleaning(df)
    return df_cleaned

df_country = get_data(selected_country)

if df_country.empty:
    st.error(f"Could not load data for {selected_country}. Please check data source.")
    st.stop() # Stop the app if no data

# Date Range Slider
min_date = df_country.index.min().date()
max_date = df_country.index.max().date()

date_range = st.sidebar.slider(
    "Select Date Range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    format="YYYY-MM-DD"
)

# Filter DataFrame by selected date range
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + timedelta(days=1, seconds=-1) # Include the whole end day

df_filtered = df_country.loc[start_dt:end_dt]

if df_filtered.empty:
    st.warning("No data available for the selected date range. Please adjust the slider.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.info("Data displayed is simulated for demonstration purposes.")

# --- Main Content Area ---

st.subheader(f"Data Overview for {selected_country} ({date_range[0]} to {date_range[1]})")
st.write(f"Displaying {len(df_filtered)} records.")
st.dataframe(df_filtered.head())

# --- EDA Sections ---

st.header("1. Summary Statistics & Missing Value Report")
st.subheader("Descriptive Statistics")
st.dataframe(df_filtered.describe().T)

st.subheader("Missing Value Report")
missing_values = df_filtered.isna().sum()
total_rows = len(df_filtered)
missing_percentage = (missing_values / total_rows) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage (%)': missing_percentage
}).sort_values(by='Missing Count', ascending=False)

st.dataframe(missing_df[missing_df['Missing Count'] > 0])

st.subheader("Columns with >5% Nulls")
high_null_cols = missing_percentage[missing_percentage > 5]
if not high_null_cols.empty:
    st.dataframe(high_null_cols.to_frame(name='Missing Percentage (%)'))
else:
    st.info("No columns with more than 5% nulls in the filtered data.")


st.header("2. Time Series Analysis")
st.markdown("Observe daily and seasonal patterns in solar irradiance and temperature.")

time_series_metrics = ['GHI', 'DNI', 'DHI', 'Tamb']
fig_ts = utils.plot_time_series(df_filtered, time_series_metrics, main_title=f'Solar Irradiance and Temperature Over Time for {selected_country}')
st.pyplot(fig_ts)
plt.close(fig_ts) # Close the figure to prevent memory issues

st.subheader("Hourly and Monthly Patterns")
df_filtered['Hour'] = df_filtered.index.hour
df_filtered['Month'] = df_filtered.index.month

fig_hourly, ax_hourly = plt.subplots(figsize=(12, 6))
sns.lineplot(x='Hour', y='GHI', data=df_filtered, errorbar=None, label='GHI', ax=ax_hourly)
sns.lineplot(x='Hour', y='Tamb', data=df_filtered, errorbar=None, label='Tamb', ax=ax_hourly)
ax_hourly.set_title('Average Daily Patterns (Hourly)')
ax_hourly.set_ylabel('Value')
ax_hourly.set_xlabel('Hour of Day')
ax_hourly.legend()
st.pyplot(fig_hourly)
plt.close(fig_hourly)

fig_monthly, ax_monthly = plt.subplots(figsize=(12, 6))
sns.boxplot(x='Month', y='GHI', data=df_filtered, ax=ax_monthly)
ax_monthly.set_title('GHI Distribution by Month')
ax_monthly.set_ylabel('GHI (W/m²)')
ax_monthly.set_xlabel('Month')
st.pyplot(fig_monthly)
plt.close(fig_monthly)


st.header("3. Cleaning Impact")
st.markdown("Visualizing the effect of outlier detection and imputation on sensor readings.")
fig_clean_impact = utils.plot_cleaning_impact(df_filtered)
st.pyplot(fig_clean_impact)
plt.close(fig_clean_impact)


st.header("4. Correlation & Relationship Analysis")
st.markdown("Understanding the relationships between different environmental and solar metrics.")
fig_corr = utils.plot_correlation_heatmap(df_filtered)
if fig_corr:
    st.pyplot(fig_corr)
    plt.close(fig_corr)
else:
    st.info("Not enough numeric columns to plot correlation heatmap.")

st.subheader("Scatter Plots: Wind, Humidity & Temperature")
fig_wind_scatter, axes_wind_scatter = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(x='WS', y='GHI', data=df_filtered, alpha=0.5, ax=axes_wind_scatter[0])
axes_wind_scatter[0].set_title('Wind Speed (WS) vs. GHI')
sns.scatterplot(x='WSgust', y='GHI', data=df_filtered, alpha=0.5, ax=axes_wind_scatter[1])
axes_wind_scatter[1].set_title('Wind Gust Speed (WSgust) vs. GHI')
sns.scatterplot(x='WD', y='GHI', data=df_filtered, alpha=0.5, ax=axes_wind_scatter[2])
axes_wind_scatter[2].set_title('Wind Direction (WD) vs. GHI')
st.pyplot(fig_wind_scatter)
plt.close(fig_wind_scatter)

fig_rh_scatter = utils.plot_temperature_humidity_relation(df_filtered)
st.pyplot(fig_rh_scatter)
plt.close(fig_rh_scatter)


st.header("5. Wind & Distribution Analysis")
st.markdown("Histograms for wind speed and direction.")
fig_wind_dist = utils.plot_wind_distribution(df_filtered)
st.pyplot(fig_wind_dist)
plt.close(fig_wind_dist)

fig_ghi_ws_hist, axes_ghi_ws_hist = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df_filtered['GHI'].dropna(), kde=True, bins=50, ax=axes_ghi_ws_hist[0])
axes_ghi_ws_hist[0].set_title('Distribution of GHI')
sns.histplot(df_filtered['WS'].dropna(), kde=True, bins=30, ax=axes_ghi_ws_hist[1])
axes_ghi_ws_hist[1].set_title('Distribution of Wind Speed (WS)')
st.pyplot(fig_ghi_ws_hist)
plt.close(fig_ghi_ws_hist)


st.header("6. Temperature Analysis")
st.markdown("Examining the influence of relative humidity on temperature and solar radiation.")
# This is covered by the scatter plots in section 4, but can be reiterated.
st.info("Observations on RH vs. Tamb and RH vs. GHI are shown in the scatter plots above.")


st.header("7. Bubble Chart: GHI vs. Tamb with RH/BP")
st.markdown("Visualize GHI vs. Ambient Temperature, with bubble size indicating Relative Humidity or Barometric Pressure.")
fig_bubble = utils.plot_bubble_chart(df_filtered)
st.pyplot(fig_bubble)
plt.close(fig_bubble)


st.header("8. Cross-Country Comparison (Overall)")
st.markdown("""
This section provides a high-level comparison of solar potential across all simulated countries.
""")

# Load all countries' data for comparison (re-using get_data for caching)
@st.cache_data
def get_all_country_data():
    all_dfs = []
    for country in utils.get_country_list():
        df_c = get_data(country) # This will call utils.load_data and utils.perform_cleaning for each country
        if not df_c.empty:
            df_c['Country'] = country # Add country column for comparison
            all_dfs.append(df_c)
    return pd.concat(all_dfs) if all_dfs else pd.DataFrame()

all_countries_combined_df = get_all_country_data()

if not all_countries_combined_df.empty:
    st.subheader("GHI, DNI, DHI Distribution Across Countries")
    for metric in ['GHI', 'DNI', 'DHI']:
        fig_boxplot_comp = utils.plot_boxplot_comparison(all_countries_combined_df, metric)
        st.pyplot(fig_boxplot_comp)
        plt.close(fig_boxplot_comp)

    st.subheader("Summary Table: Mean, Median, Std Dev of Key Metrics")
    summary_metrics = ['GHI', 'DNI', 'DHI']
    summary_table = utils.get_summary_table(all_countries_combined_df, summary_metrics)
    st.dataframe(summary_table.style.format({
        (m, 'mean'): "{:.2f}" for m in summary_metrics
    }).format({
        (m, 'median'): "{:.2f}" for m in summary_metrics
    }).format({
        (m, 'std'): "{:.2f}" for m in summary_metrics
    }))

    st.subheader("Average GHI Ranking")
    fig_avg_ghi_rank = utils.plot_average_ghi_ranking(all_countries_combined_df)
    st.pyplot(fig_avg_ghi_rank)
    plt.close(fig_avg_ghi_rank)

    st.subheader("Statistical Testing: One-way ANOVA on GHI")
    ghi_by_country_list = [all_countries_combined_df['GHI'][all_countries_combined_df['Country'] == country].dropna()
                           for country in utils.get_country_list()]

    # Filter out empty lists for ANOVA
    ghi_by_country_list = [g for g in ghi_by_country_list if not g.empty]

    if len(ghi_by_country_list) > 1 and all(len(g) > 1 for g in ghi_by_country_list):
        f_statistic, p_value = stats.f_oneway(*ghi_by_country_list)
        st.write(f"One-way ANOVA F-statistic for GHI: **{f_statistic:.2f}**")
        st.write(f"One-way ANOVA p-value for GHI: **{p_value:.4f}**")

        if p_value < 0.05:
            st.success("Interpretation: The p-value is less than 0.05, indicating a statistically significant difference in mean GHI values across the countries.")
        else:
            st.info("Interpretation: The p-value is greater than 0.05, indicating no statistically significant difference in mean GHI values across the countries.")
    else:
        st.warning("Not enough valid data points across multiple countries to perform ANOVA.")

else:
    st.warning("No combined data available for cross-country comparison.")

st.markdown("---")
st.markdown("Dashboard developed using Streamlit.")

