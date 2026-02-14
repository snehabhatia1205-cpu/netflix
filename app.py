import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.set_page_config(page_title="Netflix Content Dashboard", layout="wide")

st.title("Netflix Content Performance & Viewer Retention Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    
    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Data Cleaning
    st.subheader("Data Cleaning")
    df.drop_duplicates(inplace=True)
    df.fillna({
        'Genre': 'Unknown',
        'Director': 'Unknown',
        'Cast': 'Unknown',
        'Language': 'Unknown',
        'Country': 'Unknown'
    }, inplace=True)
    st.write("Missing values filled and duplicates removed.")

    # Regex Example: Extract first actor from 'Cast'
    st.subheader("Feature Engineering with Regex")
    df['Main_Actor'] = df['Cast'].apply(lambda x: re.split(',|;', str(x))[0])
    st.write(df[['Cast', 'Main_Actor']].head())

    # Feature Engineering
    st.subheader("Feature Engineering")
    df['Views_per_Minute'] = df['Views'] / df['Duration_Minutes']
    df['Watch_Ratio'] = df['Avg_Watch_Time_Minutes'] / df['Duration_Minutes']
    st.write(df[['Views', 'Duration_Minutes', 'Views_per_Minute', 'Watch_Ratio']].head())

    # Normalization
    st.subheader("Normalization")
    numeric_cols = ['Duration_Minutes', 'Views', 'Avg_Watch_Time_Minutes', 'Retention_Rate', 'Rating', 'Views_per_Minute', 'Watch_Ratio']
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    st.write(df[numeric_cols].head())

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    # Visualizations
    st.subheader("Visualizations")
    st.bar_chart(df.groupby('Genre')['Views'].mean())
    st.line_chart(df[['Retention_Rate', 'Rating']].head(1000))  # sample first 1000 rows for performance

else:
    st.info("Upload a CSV file to start analysis.")
