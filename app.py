import streamlit as st
import pandas as pd
import plotly.express as px
from src.twitch_utils import get_chat_dataframe
from src.model import SentimentAnalyzer
import numpy as np
import time

# Page Layout
st.set_page_config(page_title="Twitch Sentiment Analyzer", layout="wide")

st.title("💜 Twitch Sentiment & Engagement Analyzer")
st.markdown("Analyze the vibe of any Twitch VOD using AI. Powered by **RoBERTa**.")

# Sidebar
st.sidebar.header("Configuration")
model_name = st.sidebar.text_input("HuggingFace Model", value="cardiffnlp/twitter-roberta-base-sentiment-latest")
max_msgs = st.sidebar.number_input("Max Messages to Analyze", min_value=100, max_value=10000, value=1000, step=100)

@st.cache_resource
def load_model(name):
    return SentimentAnalyzer(model_name=name)

# Initialize Model
try:
    analyzer = load_model(model_name)
    st.sidebar.success("Model Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Input Section
col1, col2 = st.columns([3, 1])
with col1:
    vod_url = st.text_input("Enter Twitch VOD URL (e.g., https://www.twitch.tv/videos/123456789)")

with col2:
    analyze_btn = st.button("Analyze VOD", type="primary", use_container_width=True)

# Debug Options
use_mock = st.sidebar.checkbox("🛠 Use Mock Data (Debug)", help="Check this if download hangs")

if analyze_btn and vod_url:
    with st.status("Processing VOD...", expanded=True) as status:
        
        # Step 1: Download Chat
        st.write("Fetching chat logs...")
        download_progress = st.progress(0)
        
        def update_download_progress(count):
            # Scale progress based on max_msgs (capped at 90% until done)
            if max_msgs:
                prog = min(count / max_msgs, 0.95)
                download_progress.progress(prog, text=f"Downloaded {count} messages...")
        
        if use_mock:
            from src.twitch_utils import generate_mock_chat_dataframe
            time.sleep(1) # Simulate delay
            df = generate_mock_chat_dataframe(max_msgs)
        else:
            df = get_chat_dataframe(vod_url, max_messages=max_msgs, progress_callback=update_download_progress)
            
        download_progress.progress(1.0, text="Download Complete!")
        
        if df.empty:
            status.update(label="Failed to download chat.", state="error")
            st.error("Could not fetch chat. Check the URL or try a different VOD.")
            st.stop()
            
        st.write(f"Downloaded {len(df)} messages.")
        
        # Step 2: Analyze Sentiment
        st.write("Analyzing sentiment (this may take a moment)...")
        sentiment_progress = st.progress(0)
        
        # Batch inference with progress
        messages = df['message'].tolist()
        results = []
        batch_size = 10
        total_msgs = len(messages)
        
        for i in range(0, total_msgs, batch_size):
            batch = messages[i:i+batch_size]
            batch_results = analyzer.predict_batch(batch)
            results.extend(batch_results)
            
            # Update progress
            prog = min((i + len(batch)) / total_msgs, 1.0)
            sentiment_progress.progress(prog, text=f"Analyzed {i + len(batch)}/{total_msgs} messages")
        
        sentiment_progress.empty() # Clear progress bar
        
        # Parse results
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
        
        # Map labels to numeric for graphing: Negative=-1, Neutral=0, Positive=1
        label_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        df['sentiment_val'] = df['sentiment_label'].map(label_map)
        
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # --- Dashboard ---
    
    # KPIs
    avg_sentiment = df['sentiment_val'].mean()
    pos_count = len(df[df['sentiment_label'] == 'Positive'])
    neg_count = len(df[df['sentiment_label'] == 'Negative'])
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Average Sentiment Vibe", f"{avg_sentiment:.2f}", delta_color="normal")
    kpi2.metric("Positive Messages", pos_count, delta=f"{(pos_count/len(df))*100:.1f}%")
    kpi3.metric("Negative Messages", neg_count, delta=f"-{(neg_count/len(df))*100:.1f}%", delta_color="inverse")
    
    # Rolling Average Chart
    st.subheader("📈 Sentiment Flow Over Time")
    
    # Calculate rolling average
    window_size = int(len(df) * 0.05) if len(df) > 100 else 5
    df['rolling_sentiment'] = df['sentiment_val'].rolling(window=window_size).mean()
    
    fig = px.line(df, x='time_text', y='rolling_sentiment', title=f'Sentiment Trend (Rolling Window: {window_size} msgs)',
                  labels={'rolling_sentiment': 'Sentiment Score (-1 to 1)', 'time_text': 'Video Time'})
    fig.update_layout(yaxis_range=[-1, 1])
    # Add color regions? 
    fig.add_hrect(y0=0.2, y1=1, line_width=0, fillcolor="green", opacity=0.1)
    fig.add_hrect(y0=-1, y1=-0.2, line_width=0, fillcolor="red", opacity=0.1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Moments
    st.subheader("🔥 Top Hype Moments")
    # Identify peaks in rolling sentiment
    top_indices = df['rolling_sentiment'].nlargest(5).index
    cols = st.columns(5)
    for i, idx in enumerate(top_indices):
        if idx >= 0:
            row = df.iloc[idx]
            with cols[i]:
                st.info(f"**{row['time_text']}**\n\n{row['message']}")

    # Raw Data Explorer
    with st.expander("Explore Raw Data"):
        st.dataframe(df[['time_text', 'author', 'message', 'sentiment_label', 'sentiment_score']])
