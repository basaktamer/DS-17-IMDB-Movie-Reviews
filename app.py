import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
import os

# 1. Page Configuration - MUST BE FIRST
st.set_page_config(page_title="IMDB Sentiment Predictor", page_icon="🎬", layout="wide")

# 2. Text Cleaning Function (Matches your training logic)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text)     # Remove digits
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text

# 3. Load Assets with Error Handling
@st.cache_resource
def load_assets():
    # Load Model - using compile=False to bypass version-related metadata errors
    model = tf.keras.models.load_model('imdb_sentiment_model.keras', compile=False)
    # Re-compile manually for prediction
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Load Vectorizer
    with open('imdb_vectorizer.pkl', 'rb') as f:
        vect = pickle.load(f)
    return model, vect

try:
    model, vect = load_assets()

    st.title("🎬 IMDB Movie Review Sentiment Analysis")
    st.markdown("### Project #17: Deep Learning for NLP")

    # Creating Tabs for a professional look
    tab1, tab2 = st.tabs(["🔍 Sentiment Predictor", "📊 Dataset Insights"])

    with tab1:
        st.markdown("#### Test the Model")
        user_input = st.text_area("Enter a movie review to analyze:", 
                                  placeholder="Type your review here (e.g., 'The plot was incredible!')",
                                  height=150)

        if st.button("Analyze Sentiment"):
            if user_input.strip() != "":
                # Process the input
                cleaned_input = clean_text(user_input)
                
                # Vectorize to match the 5000-feature input layer
                vectorized_input = vect.transform([cleaned_input]).toarray()
                
                # Predict
                prediction = model.predict(vectorized_input.astype('float32'))
                score = float(prediction[0][0])
                
                # Display Results
                st.divider()
                if score > 0.5:
                    st.success(f"**Result: Positive Sentiment** (Confidence Score: {score:.2%})")
                    st.balloons()
                else:
                    st.error(f"**Result: Negative Sentiment** (Confidence Score: {1-score:.2%})")
            else:
                st.warning("Please enter some text before analyzing.")

    with tab2:
        st.header("Exploratory Data Analysis")
        st.write("Visualizing common keywords from the 50,000 review training set.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Positive Sentiment Words")
            if os.path.exists("imdb_positive_cloud.png"):
                st.image("imdb_positive_cloud.png", use_container_width=True)
            else:
                st.info("Positive Word Cloud image not found.")

        with col2:
            st.subheader("Negative Sentiment Words")
            if os.path.exists("imdb_negative_cloud.png"):
                st.image("imdb_negative_cloud.png", use_container_width=True)
            else:
                st.info("Negative Word Cloud image not found.")

except Exception as e:
    st.error(f"⚠️ Deployment Error: {e}")
    st.info("Ensure that 'imdb_sentiment_model.keras' and 'imdb_vectorizer.pkl' are uploaded and not corrupted.")

st.sidebar.markdown("---")
st.sidebar.write("Developed by **Basak**")
st.sidebar.write("Accuracy: **87.12%**")