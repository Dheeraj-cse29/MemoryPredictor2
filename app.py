import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="AI Memory Predictor", page_icon="🧠", layout="centered")

# Load and train model
@st.cache_resource
def load_model():
    data = pd.read_csv("memory.csv")
    X = data[['StudyHours', 'SleepHours', 'Revision', 'StressLevel']]
    y = data['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test) * 100
    return model, accuracy

model, accuracy = load_model()

# Session state for tracking predictions
if "remember_count" not in st.session_state:
    st.session_state.remember_count = 0
if "forget_count" not in st.session_state:
    st.session_state.forget_count = 0

# Login system
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Login - AI Memory Predictor")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid Login")
else:
    st.title("AI Memory Retention Predictor")
    st.write(f"Model Accuracy: **{accuracy:.2f}%**")

    col1, col2 = st.columns(2)
    with col1:
        study = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
        sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    with col2:
        revision = st.number_input("Revision Count", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
        stress = st.number_input("Stress Level (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=1.0)

    if st.button("Predict", type="primary"):
        result = model.predict([[study, sleep, revision, stress]])
        if result[0] == 1:
            st.success("Prediction: You will REMEMBER it!")
            st.session_state.remember_count += 1
        else:
            st.error("Prediction: You will FORGET it!")
            st.session_state.forget_count += 1

    # Graphs
    st.subheader("Charts")
    tab1, tab2 = st.tabs(["Accuracy", "Prediction Results"])

    with tab1:
        fig = go.Figure(data=[go.Bar(x=["Accuracy"], y=[accuracy], marker_color="steelblue")])
        fig.update_layout(title="Model Accuracy", yaxis_title="Percentage")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure(data=[go.Bar(
            x=["Remember", "Forget"],
            y=[st.session_state.remember_count, st.session_state.forget_count],
            marker_color=["green", "red"]
        )])
        fig2.update_layout(title="Prediction Results", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()