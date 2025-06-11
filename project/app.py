import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="FIFA Player Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the FIFA dataset"""
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/umar2926/Dataset/refs/heads/main/fifa_players.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please upload fifa_sample_data.csv")
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        with open('/project/models/trained_models.pkl', 'rb') as f:
            models = pickle.load(f)
        with open('/models/app_data.pkl', 'rb') as f:
            app_data = pickle.load(f)
        return models, app_data
    except FileNotFoundError:
        st.error("❌ Models not found! Please upload model files")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ AI-Powered FIFA Player Analytics</h1>', unsafe_allow_html=True)
    st.markdown("### 🎯 Predict Player Performance & Analyze Football Talent")
    
    # Load data and models
    df = load_data()
    models, app_data = load_models()
    
    if df is None or models is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis",
        [
            "🏠 Home Dashboard",
            "📊 Q1: Performance Factors", 
            "👤 Q2: Age Impact Analysis",
            "🌍 Q3: Nationality Rankings",
            "🦶 Q4: Foot Preference Study",
            "🤖 Q5: Predictive Analytics",
            "🔮 Player Predictor Tool"
        ]
    )
    
    # Display selected page
    if page == "🏠 Home Dashboard":
        show_dashboard(df)
    elif page == "📊 Q1: Performance Factors":
        question_1_analysis(df)
    elif page == "👤 Q2: Age Impact Analysis":
        question_2_analysis(df)
    elif page == "🌍 Q3: Nationality Rankings":
        question_3_analysis(df)
    elif page == "🦶 Q4: Foot Preference Study":
        question_4_analysis(df)
    elif page == "🤖 Q5: Predictive Analytics":
        question_5_analysis(df, models, app_data)
    elif page == "🔮 Player Predictor Tool":
        player_predictor(models, app_data)

def show_dashboard(df):
    st.markdown('<h2 class="section-header">📈 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Players", len(df))
    with col2:
        st.metric("Avg Rating", f"{df['overall_rating'].mean():.1f}")
    with col3:
        st.metric("Countries", df['nationality'].nunique() if 'nationality' in df.columns else "N/A")
    with col4:
        st.metric("Age Range", f"{df['age'].min()}-{df['age'].max()}")
    
    # Quick visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='overall_rating', title='Overall Rating Distribution', 
                          color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x='age', y='overall_rating', title='Age vs Rating',
                        color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig, use_container_width=True)

def question_1_analysis(df):
    st.markdown('<h2 class="section-header">📊 Q1: What factors influence player performance?</h2>', unsafe_allow_html=True)
    
    # Correlation analysis
    numeric_df = df.select_dtypes(include='number')
    corr_with_rating = numeric_df.corr()['overall_rating'].sort_values(ascending=False).head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Top Correlations with Overall Rating")
        fig = px.bar(x=corr_with_rating.values[1:], y=corr_with_rating.index[1:], 
                     orientation='h', title="Feature Correlations")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        top_features = corr_with_rating.head(10).index
        sns.heatmap(numeric_df[top_features].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

def question_2_analysis(df):
    st.markdown('<h2 class="section-header">👤 Q2: How does age affect performance and market value?</h2>', unsafe_allow_html=True)
    
    # Age analysis
    age_corr_rating = df['age'].corr(df['overall_rating'])
    age_corr_value = df['age'].corr(df['value_euro']) if 'value_euro' in df.columns else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Age-Rating Correlation", f"{age_corr_rating:.3f}")
    with col2:
        st.metric("Age-Value Correlation", f"{age_corr_value:.3f}")
    
    # Interactive age filter
    age_range = st.slider("Select Age Range", int(df['age'].min()), int(df['age'].max()), 
                         (int(df['age'].min()), int(df['age'].max())))
    
    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(filtered_df, x='age', y='overall_rating', 
                        title='Age vs Overall Rating', trendline='ols')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'value_euro' in df.columns:
            fig = px.scatter(filtered_df, x='age', y='value_euro', 
                            title='Age vs Market Value', trendline='ols')
            st.plotly_chart(fig, use_container_width=True)

def question_3_analysis(df):
    st.markdown('<h2 class="section-header">🌍 Q3: Which nationalities produce the highest-rated players?</h2>', unsafe_allow_html=True)
    
    if 'nationality' not in df.columns:
        st.error("Nationality data not available in this dataset")
        return
    
    # Minimum players filter
    min_players = st.slider("Minimum players per country", 5, 50, 20)
    
    player_counts = df['nationality'].value_counts()
    valid_nations = player_counts[player_counts >= min_players].index
    filtered_df = df[df['nationality'].isin(valid_nations)]
    
    top_nations = filtered_df.groupby('nationality')['overall_rating'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(x=top_nations['mean'], y=top_nations.index, 
                     orientation='h', title=f'Top 15 Countries (≥{min_players} players)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(top_nations, x='count', y='mean', 
                        hover_name=top_nations.index,
                        title='Player Count vs Average Rating')
        st.plotly_chart(fig, use_container_width=True)

def question_4_analysis(df):
    st.markdown('<h2 class="section-header">🦶 Q4: Left vs Right foot performance difference?</h2>', unsafe_allow_html=True)
    
    if 'preferred_foot' not in df.columns:
        st.error("Preferred foot data not available in this dataset")
        return
    
    df_clean = df[['preferred_foot', 'overall_rating']].dropna()
    
    left = df_clean[df_clean['preferred_foot'] == 'Left']['overall_rating']
    right = df_clean[df_clean['preferred_foot'] == 'Right']['overall_rating']
    
    # Statistical test
    t_stat, p_value = ttest_ind(left, right, equal_var=False)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("T-Statistic", f"{t_stat:.3f}")
    with col2:
        st.metric("P-Value", f"{p_value:.4f}")
    with col3:
        significance = "Significant" if p_value < 0.05 else "Not Significant"
        st.metric("Result", significance)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(df_clean, x='preferred_foot', y='overall_rating', 
                     title='Rating Distribution by Preferred Foot')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df_clean, x='overall_rating', color='preferred_foot', 
                          title='Rating Histogram by Foot Preference', opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

def question_5_analysis(df, models, app_data):
    st.markdown('<h2 class="section-header">🤖 Q5: Predictive Analytics for Recruitment</h2>', unsafe_allow_html=True)
    
    # Model performance metrics
    st.subheader("📊 Model Performance Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Random Forest R²", "0.976")
        st.metric("Random Forest RMSE", "0.910")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Linear Regression R²", "0.944")
        st.metric("Linear Regression RMSE", "1.399")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("K-Means Silhouette", "0.256")
        st.metric("Clusters", "3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature importance (if available)
    if 'random_forest' in models:
        st.subheader("🎯 Feature Importance")
        rf_model = models['random_forest']
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            features = app_data['selected_features']
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df.head(10), x='Importance', y='Feature', 
                        orientation='h', title='Top 10 Feature Importances')
            st.plotly_chart(fig, use_container_width=True)

def player_predictor(models, app_data):
    st.markdown('<h2 class="section-header">🔮 Player Performance Predictor</h2>', unsafe_allow_html=True)
    
    if models is None or app_data is None:
        st.error("Models not loaded properly")
        return
    
    st.write("### 🎯 Input Player Attributes")
    
    # Create input fields for each feature
    selected_features = app_data['selected_features']
    user_inputs = {}
    
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(selected_features):
        with col1 if i % 2 == 0 else col2:
            if feature == 'age':
                user_inputs[feature] = st.slider(f"{feature}", 17, 45, 25)
            elif 'euro' in feature.lower():
                user_inputs[feature] = st.number_input(f"{feature}", min_value=0, value=1000000)
            elif '(1-5)' in feature:
                user_inputs[feature] = st.slider(f"{feature}", 1, 5, 3)
            else:
                user_inputs[feature] = st.slider(f"{feature}", 0, 100, 50)
    
    if st.button("🚀 Predict Player Rating", type="primary"):
        # Prepare input data
        input_data = np.array([user_inputs[feature] for feature in selected_features]).reshape(1, -1)
        
        # Scale the input
        if 'scaler' in models:
            input_scaled = models['scaler'].transform(input_data)
        else:
            input_scaled = input_data
        
        # Make predictions
        rf_prediction = models['random_forest'].predict(input_data)[0]
        lr_prediction = models['linear_regression'].predict(input_data)[0]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"🌟 Random Forest Prediction: **{rf_prediction:.1f}**")
        with col2:
            st.info(f"📈 Linear Regression Prediction: **{lr_prediction:.1f}**")
        with col3:
            avg_prediction = (rf_prediction + lr_prediction) / 2
            st.warning(f"⚖️ Average Prediction: **{avg_prediction:.1f}**")
        
        # Performance interpretation
        if avg_prediction >= 80:
            st.balloons()
            st.success("🌟 **Elite Player** - This player has exceptional potential!")
        elif avg_prediction >= 70:
            st.success("⭐ **Quality Player** - This player has solid professional potential!")
        elif avg_prediction >= 60:
            st.info("📈 **Developing Player** - This player shows promise with room for growth!")
        else:
            st.warning("💪 **Work in Progress** - This player needs significant development!")

if __name__ == "__main__":
    main()
