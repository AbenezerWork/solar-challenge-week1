# app/utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plot style
sns.set_style('whitegrid')

def get_country_list():
    """Returns a list of simulated countries."""
    return ['Benin', 'Sierra Leone', 'Togo']

def load_data(country_name: str):
    """
    Simulates loading cleaned solar data for a given country.
    In a real scenario, this would load from a CSV file.
    Since no actual data is provided, this generates dummy data.
    """
    print(f"Simulating data load for {country_name}...")
    # Simulate a time range
    start_date = pd.to_datetime('2022-01-01 00:00')
    end_date = pd.to_datetime('2023-01-01 00:00')
    num_hours = int((end_date - start_date).total_seconds() / 3600)
    timestamps = pd.date_range(start=start_date, periods=num_hours, freq='H')

    # Simulate data based on country to show some variation
    np.random.seed(42) # for reproducibility

    if country_name == 'Benin':
        ghi_mean, dni_mean, dhi_mean = 240, 160, 115
        temp_mean, rh_mean = 28, 75
        wind_mean = 3
    elif country_name == 'Sierra Leone':
        ghi_mean, dni_mean, dhi_mean = 200, 110, 110
        temp_mean, rh_mean = 26, 80
        wind_mean = 2.5
    else: # Togo
        ghi_mean, dni_mean, dhi_mean = 230, 150, 115
        temp_mean, rh_mean = 27, 70
        wind_mean = 3.5

    data = {
        'Timestamp': timestamps,
        'GHI': np.maximum(0, np.random.normal(ghi_mean, 150, num_hours) * (np.sin(timestamps.hour * np.pi / 12) + 0.5)),
        'DNI': np.maximum(0, np.random.normal(dni_mean, 100, num_hours) * (np.sin(timestamps.hour * np.pi / 12) + 0.5)),
        'DHI': np.maximum(0, np.random.normal(dhi_mean, 80, num_hours) * (np.sin(timestamps.hour * np.pi / 12) + 0.5)),
        'ModA': np.maximum(0, np.random.normal(ghi_mean * 0.8, 100, num_hours) * (np.sin(timestamps.hour * np.pi / 12) + 0.5)),
        'ModB': np.maximum(0, np.random.normal(ghi_mean * 0.75, 90, num_hours) * (np.sin(timestamps.hour * np.pi / 12) + 0.5)),
        'Tamb': np.random.normal(temp_mean, 5, num_hours),
        'RH': np.random.normal(rh_mean, 10, num_hours),
        'WS': np.maximum(0, np.random.normal(wind_mean, 1.5, num_hours)),
        'WSgust': np.maximum(0, np.random.normal(wind_mean + 1, 2, num_hours)),
        'WD': np.random.randint(0, 360, num_hours),
        'BP': np.random.normal(1013, 5, num_hours),
        'Cleaning': np.random.randint(0, 2, num_hours),
        'Precipitation': np.maximum(0, np.random.normal(0.1, 0.5, num_hours)),
        'TModA': np.random.normal(temp_mean + 10, 8, num_hours),
        'TModB': np.random.normal(temp_mean + 8, 7, num_hours),
        'ModA_original': np.maximum(0, np.random.normal(ghi_mean * 0.8, 100, num_hours) * (np.sin(timestamps.hour * np.pi / 12) + 0.5)),
        'ModB_original': np.maximum(0, np.random.normal(ghi_mean * 0.75, 90, num_hours) * (np.sin(timestamps.hour * np.pi / 12) + 0.5)),
        'Cleaning_Flag': np.random.choice(['Original', 'Cleaned'], size=num_hours, p=[0.9, 0.1])
    }

    df = pd.DataFrame(data).set_index('Timestamp')
    
    # Simulate some missing values for demonstration
    for col in ['GHI', 'DNI', 'RH']:
        df.loc[df.sample(frac=0.01).index, col] = np.nan

    return df

def plot_time_series(df: pd.DataFrame, metrics: list, main_title: str = None):
    """Plots time series for specified metrics with an optional main title."""
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 2 * len(metrics)), sharex=True)
    if len(metrics) == 1: # Handle single subplot case
        axes = [axes]
    
    if main_title:
        fig.suptitle(main_title, fontsize=16, y=1.02) # Add a super title to the figure

    for i, metric in enumerate(metrics):
        if metric in df.columns:
            df[metric].plot(ax=axes[i], title=f'{metric} Over Time', color=sns.color_palette('viridis')[i])
            axes[i].set_ylabel(metric)
            axes[i].set_xlabel('')
        else:
            axes[i].set_title(f"{metric} not available")
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
    return fig

def plot_boxplot_comparison(df: pd.DataFrame, metric: str):
    """Plots boxplot for a given metric across countries."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Country', y=metric, data=df, palette='viridis', ax=ax)
    ax.set_title(f'{metric} Distribution Across Countries')
    ax.set_ylabel(f'{metric} (W/m²)')
    ax.set_xlabel('Country')
    return fig

def plot_average_ghi_ranking(df: pd.DataFrame):
    """Plots a bar chart ranking countries by average GHI."""
    avg_ghi_by_country = df.groupby('Country')['GHI'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=avg_ghi_by_country.index, y=avg_ghi_by_country.values, palette='coolwarm', ax=ax)
    ax.set_title('Average GHI by Country')
    ax.set_ylabel('Average GHI (W/m²)')
    ax.set_xlabel('Country')
    ax.set_ylim(0)
    return fig

def get_summary_table(df: pd.DataFrame, metrics: list):
    """Generates a summary table for specified metrics across countries."""
    summary_table = df.groupby('Country')[metrics].agg(['mean', 'median', 'std'])
    return summary_table

def perform_cleaning(df: pd.DataFrame):
    """
    Simulates the cleaning process (median imputation for missing values,
    Z-score outlier detection and imputation).
    """
    df_cleaned = df.copy()
    
    key_columns_for_cleaning = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust', 'Tamb', 'RH', 'BP', 'Precipitation', 'TModA', 'TModB', 'WSstdev', 'WDstdev', 'WD']
    outlier_check_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust', 'Tamb', 'RH', 'BP', 'Precipitation', 'TModA', 'TModB', 'WSstdev', 'WDstdev', 'WD']

    # Store original ModA and ModB for later comparison
    df_cleaned['ModA_original'] = df_cleaned['ModA']
    df_cleaned['ModB_original'] = df_cleaned['ModB']
    df_cleaned['Cleaning_Flag'] = 'Original'

    # Handle missing values (median imputation)
    for col in key_columns_for_cleaning:
        if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]):
            if df_cleaned[col].isnull().sum() > 0:
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)

    # Outlier Detection and Imputation
    combined_outlier_mask = pd.Series(False, index=df_cleaned.index)
    for col in outlier_check_cols:
        if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]):
            if not df_cleaned[col].dropna().empty:
                z_scores = np.abs(stats.zscore(df_cleaned[col].dropna()))
                col_outlier_mask = pd.Series(False, index=df_cleaned.index)
                col_outlier_mask[df_cleaned[col].dropna().index] = (z_scores > 3)
                combined_outlier_mask = combined_outlier_mask | col_outlier_mask

    if combined_outlier_mask.sum() > 0:
        for col in outlier_check_cols:
            if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                median_for_imputation = df_cleaned.loc[~combined_outlier_mask, col].median()
                df_cleaned.loc[combined_outlier_mask, col] = median_for_imputation
                # Ensure non-negativity for solar values
                if col in ['GHI', 'DNI', 'DHI', 'ModA', 'ModB']:
                    df_cleaned.loc[df_cleaned[col] < 0, col] = 0
        df_cleaned.loc[combined_outlier_mask, 'Cleaning_Flag'] = 'Cleaned'
    
    return df_cleaned

def plot_cleaning_impact(df_cleaned: pd.DataFrame):
    """Plots the average ModA & ModB pre/post-clean."""
    data_for_plot = []

    # Overall original averages
    data_for_plot.append({'Metric': 'ModA', 'Value': df_cleaned['ModA_original'].mean(), 'State': 'Original (Overall)'})
    data_for_plot.append({'Metric': 'ModB', 'Value': df_cleaned['ModB_original'].mean(), 'State': 'Original (Overall)'})

    # Averages for rows that were flagged as 'Cleaned'
    if 'Cleaned' in df_cleaned['Cleaning_Flag'].unique():
        cleaned_rows_df = df_cleaned[df_cleaned['Cleaning_Flag'] == 'Cleaned']
        if not cleaned_rows_df.empty:
            data_for_plot.append({'Metric': 'ModA', 'Value': cleaned_rows_df['ModA_original'].mean(), 'State': 'Original (Cleaned Rows)'})
            data_for_plot.append({'Metric': 'ModB', 'Value': cleaned_rows_df['ModB_original'].mean(), 'State': 'Original (Cleaned Rows)'})
            data_for_plot.append({'Metric': 'ModA', 'Value': cleaned_rows_df['ModA'].mean(), 'State': 'Cleaned (Cleaned Rows)'})
            data_for_plot.append({'Metric': 'ModB', 'Value': cleaned_rows_df['ModB'].mean(), 'State': 'Cleaned (Cleaned Rows)'})

    plot_df_impact = pd.DataFrame(data_for_plot)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Metric', y='Value', hue='State', data=plot_df_impact, palette='viridis', ax=ax)
    ax.set_title('Average ModA & ModB: Original vs. Cleaned (Focus on Impacted Rows)')
    ax.set_ylabel('Average Sensor Reading (W/m²)')
    ax.set_xlabel('Sensor')
    return fig

def plot_correlation_heatmap(df: pd.DataFrame):
    """Plots a heatmap of correlations for key solar metrics and temperatures."""
    correlation_cols = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB', 'Tamb', 'RH', 'WS', 'BP']
    correlation_cols = [col for col in correlation_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if len(correlation_cols) > 1:
        correlation_matrix = df[correlation_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title('Correlation Heatmap of Key Variables')
        return fig
    else:
        return None

def plot_wind_distribution(df: pd.DataFrame):
    """Plots histograms for Wind Speed and Wind Direction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df['WS'].dropna(), kde=True, bins=30, ax=axes[0])
    axes[0].set_title('Distribution of Wind Speed (WS)')
    axes[0].set_xlabel('Wind Speed (m/s)')
    axes[0].set_ylabel('Frequency')

    sns.histplot(df['WD'].dropna(), bins=36, kde=False, ax=axes[1])
    axes[1].set_title('Distribution of Wind Direction (WD)')
    axes[1].set_xlabel('Wind Direction (°N (to east))')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    return fig

def plot_temperature_humidity_relation(df: pd.DataFrame):
    """Plots RH vs. Tamb and RH vs. GHI with regression lines."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.regplot(x='RH', y='Tamb', data=df, scatter_kws={'alpha':0.3}, ax=axes[0])
    axes[0].set_title('Relative Humidity (RH) vs. Ambient Temperature (Tamb)')
    axes[0].set_xlabel('Relative Humidity (%)')
    axes[0].set_ylabel('Ambient Temperature (°C)')

    sns.regplot(x='RH', y='GHI', data=df, scatter_kws={'alpha':0.3}, ax=axes[1])
    axes[1].set_title('Relative Humidity (RH) vs. GHI')
    axes[1].set_xlabel('Relative Humidity (%)')
    axes[1].set_ylabel('GHI (W/m²)')

    plt.tight_layout()
    return fig

def plot_bubble_chart(df: pd.DataFrame):
    """Plots GHI vs. Tamb with bubble size based on RH or BP."""
    bubble_size_col = 'RH'
    if 'BP' in df.columns and not df['BP'].isnull().all() and (df['BP'] >= 0).all():
        bubble_size_col = 'BP'

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='Tamb', y='GHI', size=bubble_size_col, hue=bubble_size_col,
                    data=df, sizes=(20, 2000), alpha=0.6, palette='viridis', ax=ax)
    ax.set_title(f'GHI vs. Ambient Temperature (Bubble Size = {bubble_size_col})')
    ax.set_xlabel('Ambient Temperature (°C)')
    ax.set_ylabel('GHI (W/m²)')
    ax.legend(title=bubble_size_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    return fig

