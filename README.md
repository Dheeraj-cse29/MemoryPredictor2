# AI Memory Retention Predictor

A machine-learning-based web application that helps students understand
and improve their study habits by predicting memory retention from key
learning-related factors.

## Project Overview

The **AI Memory Retention Predictor** takes information such as study
hours, sleep hours, revision count, and stress level as input. It
processes these values using a machine learning model and predicts
whether the student is more likely to **Remember** or **Forget** the
studied material.

Along with the prediction, the application provides suggestions to
improve learning habits and includes visualizations and additional
student-focused features.

## Features

-   **Admin and Student Login:** Separate login options for
    administrators and students.
-   **Memory Retention Prediction:** Predicts a Remember/Forget outcome
    from the entered study-related details.
-   **Recommendation System:** Provides suggestions based on the user's
    inputs to help improve memory retention.
-   **Accuracy Graph:** Displays model accuracy information.
-   **Result Graph:** Presents prediction-related results visually.
-   **Goal Setting System:** Allows users to set a target retention
    percentage (for example, 80%) and receive suggestions to help work
    towards it.
-   **MCQ-Based Confidence Evaluation:** Lets students select a subject,
    answer multiple-choice questions, view their score, and receive a
    Low, Medium, or High confidence level.

## Input Parameters

The prediction module uses the following inputs:

  Input            Description
  ---------------- ------------------------------------------
  Study Hours      Number of hours spent studying
  Sleep Hours      Number of hours slept
  Revision Count   Number of times the material was revised
  Stress Level     User's stress level

## Technologies Used

-   **Python** -- application and machine-learning logic
-   **Streamlit** -- interactive Python web application
-   **Scikit-learn** -- machine-learning functionality
-   **Pandas** -- data handling
-   **Matplotlib** -- graph visualizations
-   **GitHub** -- source-code version control
-   **Streamlit Community Cloud** -- deployment of the Streamlit
    application
-   **Vercel** -- deployment of the frontend UI, where configured

> Update this list if your final implementation uses different libraries
> or deployment arrangements.

## Project Structure

``` text
MemoryPredictor2/
├── app.py
├── memory.csv
├── requirements.txt
└── README.md
```

-   `app.py` -- main application code.
-   `memory.csv` -- dataset used by the project.
-   `requirements.txt` -- Python package dependencies.
-   `README.md` -- project documentation.

## Getting Started

### Prerequisites

-   Python 3.9 or a compatible version
-   pip (Python package installer)
-   Git

### Installation

1.  Clone the repository:

    ``` bash
    git clone https://github.com/YOUR-USERNAME/MemoryPredictor2.git
    ```

2.  Open the project folder:

    ``` bash
    cd MemoryPredictor2
    ```

3.  Install the required packages:

    ``` bash
    pip install -r requirements.txt
    ```

4.  Run the Streamlit application:

    ``` bash
    streamlit run app.py
    ```

5.  Open the local URL shown in the terminal (usually
    `http://localhost:8501`).

## Deployment

The project source code is maintained in GitHub. The Streamlit
application can be deployed through **Streamlit Community Cloud** by
connecting the GitHub repository, selecting the branch, and specifying
`app.py` as the main file.

If the project has a separate frontend, that UI can be deployed through
**Vercel** using the corresponding frontend repository and build
configuration.

## How to Use

1.  Open the deployed application.
2.  Choose the appropriate Admin or Student login option.
3.  Enter the required study details.
4.  Select **Predict** to view the memory-retention prediction.
5.  Review the recommendations provided by the system.
6.  Explore the accuracy and result graphs.
7.  Use the Goal Setting System to set a target retention percentage.
8.  Choose a subject and complete the MCQ-based confidence evaluation.

## Purpose

The project demonstrates how machine learning and web technologies can
be applied to an educational use case. It aims to help students reflect
on study-related factors and make informed improvements to their
learning routines.

## Future Enhancements

-   Use larger and more diverse datasets to improve model
    generalization.
-   Add personalized progress tracking over time.
-   Expand the MCQ question bank and subject coverage.
-   Improve the recommendation system with more personalized
    suggestions.
-   Add additional analytics and reporting features.

## Disclaimer

The prediction is an educational project output and should not be
treated as a medical or psychological assessment. Results depend on the
dataset and model used in the implementation.

## Author

**Dheeraj**

------------------------------------------------------------------------

If you use or modify this project, please ensure that dataset sources,
model details, and third-party resources are acknowledged appropriately.
