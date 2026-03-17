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

# MCQ Database
MCQ_DATABASE = {
    "Machine Learning": [
        {
            "question": "What is the primary goal of machine learning?",
            "options": ["To create algorithms that learn from data", "To manually program all rules", "To store data", "To delete data"],
            "correct": 0
        },
        {
            "question": "Which algorithm is best for classification?",
            "options": ["Linear Regression", "K-Means", "Decision Trees", "Principal Component Analysis"],
            "correct": 2
        },
        {
            "question": "What is overfitting?",
            "options": ["Model works well on training but poorly on test data", "Model doesn't learn anything", "Using too little data", "Using only one feature"],
            "correct": 0
        },
        {
            "question": "What does normalization do?",
            "options": ["Makes data random", "Scales features to a similar range", "Removes outliers", "Increases data size"],
            "correct": 1
        },
        {
            "question": "What is cross-validation used for?",
            "options": ["To store data", "To evaluate model performance reliably", "To clean data", "To visualize data"],
            "correct": 1
        }
    ],
    "Database Management Systems": [
        {
            "question": "What does ACID stand for?",
            "options": ["Availability, Consistency, Integrity, Durability", "Atomicity, Consistency, Isolation, Durability", "Accuracy, Control, Integration, Data", "None of the above"],
            "correct": 1
        },
        {
            "question": "Which is a NoSQL database?",
            "options": ["PostgreSQL", "MySQL", "MongoDB", "Oracle"],
            "correct": 2
        },
        {
            "question": "What is a primary key?",
            "options": ["A key that opens databases", "A unique identifier for a row", "A password", "A type of encryption"],
            "correct": 1
        },
        {
            "question": "What does normalization do in databases?",
            "options": ["Makes data random", "Eliminates redundancy and improves data integrity", "Deletes duplicate data", "Encrypts data"],
            "correct": 1
        },
        {
            "question": "What is an index in a database?",
            "options": ["A list of tables", "A structure that improves query speed", "A backup mechanism", "A data type"],
            "correct": 1
        }
    ],
    "Data Structures": [
        {
            "question": "Which data structure uses LIFO principle?",
            "options": ["Queue", "Stack", "Array", "Tree"],
            "correct": 1
        },
        {
            "question": "What is the time complexity of binary search?",
            "options": ["O(n)", "O(log n)", "O(n^2)", "O(1)"],
            "correct": 1
        },
        {
            "question": "Which data structure is best for implementing a priority queue?",
            "options": ["Array", "Linked List", "Heap", "Hash Table"],
            "correct": 2
        },
        {
            "question": "What is a hash collision?",
            "options": ["When two keys hash to the same value", "When a hash function fails", "When memory overflows", "When two programs run together"],
            "correct": 0
        },
        {
            "question": "Which is NOT a linear data structure?",
            "options": ["Array", "Linked List", "Queue", "Graph"],
            "correct": 3
        }
    ]
}

def calculate_confidence_level(score):
    """Calculate confidence level based on quiz score"""
    if score >= 80:
        return "High", "#84fab0"
    elif score >= 60:
        return "Medium", "#ffd700"
    else:
        return "Low", "#fa709a"

def get_improvement_suggestions(study, sleep, revision, stress, target):
    """Generate improvement suggestions based on gap between current and target"""
    suggestions = []
    current_prediction = model.predict([[study, sleep, revision, stress]])[0]
    current_retention = 75 if current_prediction == 1 else 40
    gap = target - current_retention
    
    if gap > 0:
        if study < 5:
            suggestions.append("Increase daily study hours to 5-7 hours for better retention")
        if sleep < 7:
            suggestions.append("Aim for 7-9 hours of sleep daily to consolidate memories")
        if revision < 3:
            suggestions.append("Increase revision count to at least 3 times for better recall")
        if stress > 6:
            suggestions.append("Work on stress management techniques like meditation or exercise")
    
    return suggestions if suggestions else ["You're on track! Keep maintaining your current routine."]

# Initialize session state
if "remember_count" not in st.session_state:
    st.session_state.remember_count = 0
if "forget_count" not in st.session_state:
    st.session_state.forget_count = 0
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "retention_goal" not in st.session_state:
    st.session_state.retention_goal = None
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Predictor"

# Login Section
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="header-title">Memory Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="header-subtitle">AI-Powered Memory Retention Analysis</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.container():
            st.subheader("Login")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            if st.button("Login", use_container_width=True, type="primary"):
                if username == "admin" and password == "1234":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        
        st.markdown("---")
        st.info("Demo Credentials:\n\nUsername: admin\n\nPassword: 1234")

else:
    # Header
    st.markdown('<div class="header-title">Memory Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">AI-Powered Memory Retention Analysis</div>', unsafe_allow_html=True)
    
    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Predictor", "Goal Setting", "Quiz", "Analytics"])
    
    # PREDICTOR TAB
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        # Main Prediction Section
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("Prediction Inputs")
            
            col_input1, col_input2 = st.columns(2)
            
            with col_input1:
                st.markdown('<p class="metric-label">Study Hours (0-24)</p>', unsafe_allow_html=True)
                study = st.slider("Study Hours", 0.0, 24.0, 5.0, 0.5, label_visibility="collapsed")
                
                st.markdown('<p class="metric-label">Sleep Hours (0-24)</p>', unsafe_allow_html=True)
                sleep = st.slider("Sleep Hours", 0.0, 24.0, 7.0, 0.5, label_visibility="collapsed")
            
            with col_input2:
                st.markdown('<p class="metric-label">Revision Count (0-20)</p>', unsafe_allow_html=True)
                revision = st.slider("Revision Count", 0.0, 20.0, 2.0, 1.0, label_visibility="collapsed")
                
                st.markdown('<p class="metric-label">Stress Level (1-10)</p>', unsafe_allow_html=True)
                stress = st.slider("Stress Level", 1.0, 10.0, 5.0, 1.0, label_visibility="collapsed")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Prediction Button
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("Predict Memory Retention", use_container_width=True, type="primary"):
                    result = model.predict([[study, sleep, revision, stress]])
                    
                    if result[0] == 1:
                        st.session_state.remember_count += 1
                        st.markdown("""
                            <div class="prediction-success">
                            You will REMEMBER it!
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.session_state.forget_count += 1
                        st.markdown("""
                            <div class="prediction-fail">
                            You will FORGET it!
                            </div>
                        """, unsafe_allow_html=True)
        
        # Sidebar Metrics
        with col2:
            st.markdown("### Overview")
            
            metric_col = st.columns(1)[0]
            with metric_col:
                st.metric("Model Accuracy", f"{accuracy:.1f}%", delta="AI Powered")
                st.metric("Total Predictions", st.session_state.remember_count + st.session_state.forget_count)
                st.metric("Remember", st.session_state.remember_count)
                st.metric("Forget", st.session_state.forget_count)
        
        st.markdown("---")
        
        # Tips Section
        st.subheader("Tips for Better Memory Retention")
        
        tip_col1, tip_col2, tip_col3 = st.columns(3)
        
        with tip_col1:
            st.markdown("""
            <div class="tips-section">
            <h3>Study Smarter</h3>
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
            <h3>Sleep Well</h3>
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
            <h3>Review Often</h3>
            <ul>
            <li>Revise regularly</li>
            <li>Spaced repetition</li>
            <li>Test yourself</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # GOAL SETTING TAB
    with tab2:
        st.subheader("Target Your Memory Retention Goal")
        
        col_goal1, col_goal2 = st.columns([2, 1])
        
        with col_goal1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.write("Set your target memory retention percentage and get personalized improvement suggestions.")
            
            goal_percentage = st.slider("Target Retention %", 40, 100, 80, 5)
            
            col_input1, col_input2 = st.columns(2)
            with col_input1:
                st.markdown('<p class="metric-label">Current Study Hours</p>', unsafe_allow_html=True)
                goal_study = st.slider("Study Hours", 0.0, 24.0, 4.0, 0.5, label_visibility="collapsed", key="goal_study")
                
                st.markdown('<p class="metric-label">Current Sleep Hours</p>', unsafe_allow_html=True)
                goal_sleep = st.slider("Sleep Hours", 0.0, 24.0, 6.0, 0.5, label_visibility="collapsed", key="goal_sleep")
            
            with col_input2:
                st.markdown('<p class="metric-label">Current Revisions</p>', unsafe_allow_html=True)
                goal_revision = st.slider("Revision Count", 0.0, 20.0, 1.0, 1.0, label_visibility="collapsed", key="goal_revision")
                
                st.markdown('<p class="metric-label">Current Stress</p>', unsafe_allow_html=True)
                goal_stress = st.slider("Stress Level", 1.0, 10.0, 7.0, 1.0, label_visibility="collapsed", key="goal_stress")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("Generate Improvement Plan", use_container_width=True, type="primary"):
                st.session_state.retention_goal = goal_percentage
                suggestions = get_improvement_suggestions(goal_study, goal_sleep, goal_revision, goal_stress, goal_percentage)
                
                st.markdown(f"""
                <div class="input-section">
                <h3>Your Personalized Improvement Plan</h3>
                <p><strong>Goal:</strong> {goal_percentage}% Memory Retention</p>
                </div>
                """, unsafe_allow_html=True)
                
                for i, suggestion in enumerate(suggestions, 1):
                    st.success(f"✓ {suggestion}")
        
        with col_goal2:
            st.markdown("### Goal Progress")
            st.metric("Target Goal", f"{goal_percentage}%")
            if st.session_state.retention_goal:
                st.metric("Your Goal", f"{st.session_state.retention_goal}%")
    
    # QUIZ TAB
    with tab3:
        st.subheader("Test Your Knowledge & Confidence")
        
        subject = st.selectbox("Select Subject", list(MCQ_DATABASE.keys()))
        
        if st.button("Start Quiz", use_container_width=True, type="primary"):
            st.session_state.current_quiz = subject
            st.session_state.quiz_answers = [None] * len(MCQ_DATABASE[subject])
            st.session_state.quiz_submitted = False
        
        if "current_quiz" in st.session_state and st.session_state.current_quiz == subject:
            st.markdown("---")
            questions = MCQ_DATABASE[subject]
            
            for i, q in enumerate(questions):
                st.markdown(f"### Question {i+1}: {q['question']}")
                
                answer = st.radio(
                    f"Select answer for Q{i+1}",
                    options=q['options'],
                    key=f"q_{i}",
                    label_visibility="collapsed"
                )
                
                if answer:
                    st.session_state.quiz_answers[i] = q['options'].index(answer)
            
            col_submit1, col_submit2 = st.columns([1, 1])
            with col_submit1:
                if st.button("Submit Quiz", use_container_width=True, type="primary"):
                    correct = sum(1 for i, q in enumerate(questions) if st.session_state.quiz_answers[i] == q['correct'])
                    score = (correct / len(questions)) * 100
                    st.session_state.quiz_score = score
                    st.session_state.quiz_submitted = True
            
            if st.session_state.quiz_submitted:
                st.markdown("---")
                score = st.session_state.quiz_score
                confidence, color = calculate_confidence_level(score)
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color} 0%, {color} 100%); padding: 25px; border-radius: 12px; color: white; text-align: center;">
                <h2>Quiz Score: {score:.1f}%</h2>
                <h3>Confidence Level: <strong>{confidence}</strong></h3>
                </div>
                """, unsafe_allow_html=True)
                
                questions = MCQ_DATABASE[subject]
                for i, q in enumerate(questions):
                    if st.session_state.quiz_answers[i] == q['correct']:
                        st.success(f"Q{i+1}: Correct")
                    else:
                        st.error(f"Q{i+1}: Incorrect - Correct answer: {q['options'][q['correct']]}")
    
    # ANALYTICS TAB
    with tab4:
        st.subheader("Your Learning Analytics")
        
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
                st.info("Make predictions to see your results!")
        
        if st.session_state.quiz_score is not None:
            st.markdown("---")
            quiz_score = st.session_state.quiz_score
            confidence, _ = calculate_confidence_level(quiz_score)
            
            fig_quiz = go.Figure(data=[
                go.Bar(
                    x=["Quiz Score"],
                    y=[quiz_score],
                    marker=dict(color=["#667eea"]),
                    text=[f"{quiz_score:.1f}%"],
                    textposition="outside"
                )
            ])
            fig_quiz.update_layout(
                title=f"Latest Quiz Performance - Confidence: {confidence}",
                yaxis_title="Score (%)",
                height=400,
                showlegend=False,
                template="plotly_white",
                font=dict(size=12)
            )
            st.plotly_chart(fig_quiz, use_container_width=True)
    
    st.markdown("---")
    
    # Logout Button
    col_logout1, col_logout2, col_logout3 = st.columns([1, 2, 1])
    with col_logout2:
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
