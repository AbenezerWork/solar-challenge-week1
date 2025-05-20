import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/ghi_data.csv")

df = load_data()

# Sidebar - Country selection
st.sidebar.title("Country Selection")
countries = st.sidebar.multiselect("Select countries", df["Country"].unique())

# Sidebar - Plot type
plot_type = st.sidebar.selectbox("Select Plot Type", ["Boxplot", "Histogram", "Bar Chart"])

# Main area
st.title("Global Hunger Index Dashboard")

# Filter by country
filtered_df = df[df["Country"].isin(countries)] if countries else df

# Show plot
if plot_type == "Boxplot":
    st.subheader("Boxplot of GHI Scores")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x="GHI_Score", y="Region", ax=ax)
    st.pyplot(fig)

# Top regions table
st.subheader("Top Regions by GHI Score")
top_regions = filtered_df.groupby("Region")["GHI_Score"].mean().sort_values().head(5)
st.table(top_regions.reset_index().rename(columns={"GHI_Score": "Average GHI Score"}))

