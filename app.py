import streamlit as st
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Netflix Analytics Dashboard", layout="wide")
st.title("🎬 Netflix Content Performance & Viewer Retention Dashboard")

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload your Netflix CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # -----------------------------
    # Data Cleaning
    # -----------------------------
    df.drop_duplicates(inplace=True)
    df.fillna({
        'Genre': 'Unknown',
        'Director': 'Unknown',
        'Cast': 'Unknown',
        'Language': 'Unknown',
        'Country': 'Unknown'
    }, inplace=True)

    # Regex: extract first actor
    df['Main_Actor'] = df['Cast'].apply(lambda x: re.split(',|;', str(x))[0])
    
    # Feature Engineering
    df['Views_per_Minute'] = df['Views'] / df['Duration_Minutes']
    df['Watch_Ratio'] = df['Avg_Watch_Time_Minutes'] / df['Duration_Minutes']

    # Normalization
    numeric_cols = ['Duration_Minutes', 'Views', 'Avg_Watch_Time_Minutes', 
                    'Retention_Rate', 'Rating', 'Views_per_Minute', 'Watch_Ratio']
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    st.sidebar.header("Filters")
    genre_filter = st.sidebar.multiselect("Select Genre(s)", options=df['Genre'].unique(), default=df['Genre'].unique())
    year_filter = st.sidebar.slider("Select Release Year Range", int(df['Release_Year'].min()), int(df['Release_Year'].max()), (int(df['Release_Year'].min()), int(df['Release_Year'].max())))
    director_filter = st.sidebar.multiselect("Select Director(s)", options=df['Director'].unique(), default=df['Director'].unique())
    language_filter = st.sidebar.multiselect("Select Language(s)", options=df['Language'].unique(), default=df['Language'].unique())
    
    filtered_df = df[
        (df['Genre'].isin(genre_filter)) &
        (df['Release_Year'].between(year_filter[0], year_filter[1])) &
        (df['Director'].isin(director_filter)) &
        (df['Language'].isin(language_filter))
    ]
    
    # -----------------------------
    # KPI CARDS
    # -----------------------------
    st.subheader("📊 Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Views (scaled)", int(filtered_df['Views'].sum()))
    col2.metric("Avg Retention Rate", f"{filtered_df['Retention_Rate'].mean():.2f}")
    col3.metric("Avg Rating", f"{filtered_df['Rating'].mean():.2f}")
    top_genre = filtered_df.groupby('Genre')['Views'].mean().idxmax()
    col4.metric("Top Genre by Views", top_genre)
    
    # -----------------------------
    # Dashboard Visualizations
    # -----------------------------
    st.subheader("📈 Visualizations")

    # Genre vs Average Views
    st.write("**Average Views by Genre**")
    genre_views = filtered_df.groupby('Genre')['Views'].mean().sort_values(ascending=False)
    st.bar_chart(genre_views)

    # Retention vs Rating Scatter
    st.write("**Retention Rate vs Rating**")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=filtered_df, x='Rating', y='Retention_Rate', hue='Genre', alpha=0.7, ax=ax1)
    st.pyplot(fig1)

    # Views per Minute vs Watch Ratio
    st.write("**Views per Minute vs Watch Ratio**")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=filtered_df, x='Views_per_Minute', y='Watch_Ratio', hue='Genre', alpha=0.7, ax=ax2)
    st.pyplot(fig2)

    # Correlation Heatmap
    st.write("**Correlation Heatmap**")
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)
    
    # -----------------------------
    # Suggestions / Insights
    # -----------------------------
    st.subheader("💡 Suggestions / Insights")
    suggestions = []

    # Top retention content
    top_retention = filtered_df.sort_values(by='Retention_Rate', ascending=False).iloc[0]
    suggestions.append(f"✅ '{top_retention['Title']}' is performing best in terms of retention.")

    # Genre trends
    high_retention_genre = filtered_df.groupby('Genre')['Retention_Rate'].mean().idxmax()
    suggestions.append(f"🎯 Focus on '{high_retention_genre}' genre as it has the highest average retention.")

    # Director trends
    top_director = filtered_df.groupby('Director')['Retention_Rate'].mean().idxmax()
    suggestions.append(f"🎬 Consider content from director '{top_director}' for higher retention.")

    # Duration insights
    short_content = filtered_df[filtered_df['Duration_Minutes'] < 0.3].shape[0]  # scaled < 0.3
    suggestions.append(f"⏱ Short content (<30% normalized duration) count: {short_content}. Short content tends to retain viewers faster.")

    # Display suggestions
    for s in suggestions:
        st.info(s)

else:
    st.info("Upload a CSV file to start your Netflix analysis dashboard.")
