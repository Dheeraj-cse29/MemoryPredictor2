import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="AI Memory Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9ff;
    }
    
    .header-title {
        color: #667eea;
        text-align: center;
        font-size: 2.8em;
        font-weight: 900;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    
    .header-subtitle {
        color: #764ba2;
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 30px;
        font-weight: 500;
    }
    
    .metric-label {
        font-weight: 600;
        color: #667eea;
        font-size: 0.95em;
    }
    
    .input-section {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
        margin-bottom: 20px;
    }
    
    .prediction-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 1.3em;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(132, 250, 176, 0.3);
    }
    
    .prediction-fail {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 1.3em;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(250, 112, 154, 0.3);
    }
    
    .tips-section {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stats-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    h1, h2, h3 {
        color: #333;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stSlider {
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and train model
@st.cache_resource
def load_model():
    data = pd.read_csv("memory.csv")
    X = data[['StudyHours', 'SleepHours', 'Revision', 'StressLevel']]
    y = data['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test) * 100
    return model, accuracy

model, accuracy = load_model()

# Initialize session state
if "remember_count" not in st.session_state:
    st.session_state.remember_count = 0
if "forget_count" not in st.session_state:
    st.session_state.forget_count = 0
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login Section
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="header-title">🧠 Memory Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="header-subtitle">AI-Powered Memory Retention Analysis</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.container():
            st.subheader("🔐 Login")
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔐 Password", type="password", placeholder="Enter your password")
            
            if st.button("✓ Login", use_container_width=True, type="primary"):
                if username == "admin" and password == "1234":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
        
        st.markdown("---")
        st.info("📝 **Demo Credentials:**\n\nUsername: `admin`\n\nPassword: `1234`")

else:
    # Header
    st.markdown('<div class="header-title">🧠 Memory Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">AI-Powered Memory Retention Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    # Main Prediction Section
    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("📊 Prediction Inputs")
        
        col_input1, col_input2 = st.columns(2)
        
        with col_input1:
            st.markdown('<p class="metric-label">📚 Study Hours (0-24)</p>', unsafe_allow_html=True)
            study = st.slider("Study Hours", 0.0, 24.0, 5.0, 0.5, label_visibility="collapsed")
            
            st.markdown('<p class="metric-label">💤 Sleep Hours (0-24)</p>', unsafe_allow_html=True)
            sleep = st.slider("Sleep Hours", 0.0, 24.0, 7.0, 0.5, label_visibility="collapsed")
        
        with col_input2:
            st.markdown('<p class="metric-label">🔄 Revision Count (0-20)</p>', unsafe_allow_html=True)
            revision = st.slider("Revision Count", 0.0, 20.0, 2.0, 1.0, label_visibility="collapsed")
            
            st.markdown('<p class="metric-label">😰 Stress Level (1-10)</p>', unsafe_allow_html=True)
            stress = st.slider("Stress Level", 1.0, 10.0, 5.0, 1.0, label_visibility="collapsed")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction Button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🎯 Predict Memory Retention", use_container_width=True, type="primary"):
                result = model.predict([[study, sleep, revision, stress]])
                
                if result[0] == 1:
                    st.session_state.remember_count += 1
                    st.markdown("""
                        <div class="prediction-success">
                        ✅ You will REMEMBER it!
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.session_state.forget_count += 1
                    st.markdown("""
                        <div class="prediction-fail">
                        ❌ You will FORGET it!
                        </div>
                    """, unsafe_allow_html=True)
    
    # Sidebar Metrics
    with col2:
        st.markdown("### 📊 Overview")
        
        metric_col = st.columns(1)[0]
        with metric_col:
            st.metric("Model Accuracy", f"{accuracy:.1f}%", delta="AI Powered")
            st.metric("Total Predictions", st.session_state.remember_count + st.session_state.forget_count)
            st.metric("Remember", st.session_state.remember_count)
            st.metric("Forget", st.session_state.forget_count)
    
    st.markdown("---")
    
    # Analytics Section
    st.subheader("📈 Analytics")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig_accuracy = go.Figure(data=[
            go.Bar(
                x=["Model Accuracy"],
                y=[accuracy],
                marker=dict(color=["#667eea"]),
                text=[f"{accuracy:.1f}%"],
                textposition="outside"
            )
        ])
        fig_accuracy.update_layout(
            title="Model Accuracy Score",
            yaxis_title="Accuracy (%)",
            height=400,
            showlegend=False,
            template="plotly_white",
            font=dict(size=12)
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with chart_col2:
        total = st.session_state.remember_count + st.session_state.forget_count
        if total > 0:
            fig_results = go.Figure(data=[
                go.Bar(
                    x=["Remember", "Forget"],
                    y=[st.session_state.remember_count, st.session_state.forget_count],
                    marker=dict(color=["#84fab0", "#fa709a"]),
                    text=[st.session_state.remember_count, st.session_state.forget_count],
                    textposition="outside"
                )
            ])
            fig_results.update_layout(
                title="Your Prediction Results",
                yaxis_title="Count",
                height=400,
                showlegend=False,
                template="plotly_white",
                font=dict(size=12)
            )
            st.plotly_chart(fig_results, use_container_width=True)
        else:
            st.info("📊 Make predictions to see your results!")
    
    st.markdown("---")
    
    # Tips Section
    st.subheader("💡 Tips for Better Memory Retention")
    
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    
    with tip_col1:
        st.markdown("""
        <div class="tips-section">
        <h3>📚 Study Smarter</h3>
        <ul>
        <li>Study 5-7 hours daily</li>
        <li>Use active recall</li>
        <li>Break into chunks</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tip_col2:
        st.markdown("""
        <div class="tips-section">
        <h3>😴 Sleep Well</h3>
        <ul>
        <li>7-9 hours daily</li>
        <li>Consistent schedule</li>
        <li>Avoid screens before bed</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tip_col3:
        st.markdown("""
        <div class="tips-section">
        <h3>🔄 Review Often</h3>
        <ul>
        <li>Revise regularly</li>
        <li>Spaced repetition</li>
        <li>Test yourself</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Logout Button
    col_logout1, col_logout2, col_logout3 = st.columns([1, 2, 1])
    with col_logout2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
    
    st.markdown("""
    <div style="text-align: center; color: #999; margin-top: 30px; font-size: 0.9em;">
    <p>AI Memory Predictor | Powered by Machine Learning | Enhanced Learning</p>
    </div>
    """, unsafe_allow_html=True)
