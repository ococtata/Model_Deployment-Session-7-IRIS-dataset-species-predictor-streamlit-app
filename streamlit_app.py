import streamlit as st
import numpy as np
import pickle

@st.cache_resource
def load_models():
    with open('random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return rf_model, label_encoder

try:
    rf_model, label_encoder = load_models() 
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    models_loaded = False
    

st.title("🌷 Iris Flower Species Prediction App")
st.write("This application predicts species of flower using Random Forest Model.")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page:", ["Home", "Species Prediction"])

if page == "Home":
    st.header("Welcome to the Iris Flower Species Prediction App")
    
    st.write("""
    ### About the App
    This app helps predict house prices based on the following factors:
    
    - sepal_length	
    - sepal_width	
    - petal_length	
    - petal_width
    
    The app uses a Random Forest machine learning model.
    """)
     
    st.write("""
    ### How to Use the App
    Use the sidebar on the left to navigate between pages.
    """)

elif page == "Species Prediction":
    st.header("🌷 Iris Flower Species Prediction")
    
    if models_loaded:
        col1, col2 = st.columns(2)
        
        with col1:
            sepal_length = st.slider("Sepal Length (cm):", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            sepal_width = st.slider("Sepal Width (cm):", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        
        with col2:
            petal_length = st.slider("Petal Length (cm):", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            petal_width = st.slider("Petal Width (cm):", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        
        if st.button("Predict Species"):
            prediction = rf_model.predict(features)[0]
            species_name = label_encoder.inverse_transform([prediction])[0]

            species_images = {
                "Iris-setosa": "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcSLylr-l8efAszpjFlLQDYGXjhnVs7PAEGJqesxy36bgXHSG04ACjYwsgrIYEVVEO4P01FFNvuW13UOZjlQTZwhjQ",
                "Iris-versicolor": "https://upload.wikimedia.org/wikipedia/commons/2/27/Blue_Flag%2C_Ottawa.jpg",
                "Iris-virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
            }
            
            st.success(f"🌷Predicted Species: {species_name}")

            if species_name in species_images:
                st.image(species_images[species_name], caption=species_name, use_container_width=True)