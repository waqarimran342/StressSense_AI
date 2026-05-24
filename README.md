# StressSense AI

## A Rule-Based Expert System for Mental Stress Analysis

StressSense AI is an intelligent rule-based expert system developed to analyze and assess mental stress levels using artificial intelligence techniques such as forward chaining, certainty factors, and knowledge-based reasoning.

The system collects user symptoms and behavioral indicators, applies predefined expert rules, and generates stress assessments along with explainable recommendations.

---

# Features

- Rule-Based Expert System
- Forward Chaining Inference Engine
- Certainty Factor Calculations
- Explainable AI (Why/How Explanations)
- Stress Level Assessment
- Interactive Streamlit Web Interface
- Knowledge Base with 60+ Rules
- Modular and Scalable Architecture

---

# Technologies Used

- Python
- Experta / PyKnow
- Streamlit
- Pandas
- Plotly
- Matplotlib

---

# Project Structure

```plaintext
stresssense_ai/
│
├── knowledge_base/
│   ├── rules.py
│   ├── facts.py
│   └── ontology.py
│
├── inference/
│   ├── engine.py
│   ├── certainty_factors.py
│   └── conflict_resolution.py
│
├── explanation/
│   └── explainer.py
│
├── interface/
│   └── app.py
│
├── testing/
│   └── test_scenarios.py
│
├── docs/
│   └── knowledge_acquisition.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# System Architecture

The project consists of the following main components:

## 1. Knowledge Base
Contains:
- Facts
- Rules
- Stress Ontology
- Expert Knowledge

## 2. Inference Engine
Responsible for:
- Forward chaining
- Rule matching
- Conflict resolution
- Decision generation

## 3. Certainty Factor Module
Calculates confidence levels for:
- Stress severity
- Rule certainty
- Recommendation confidence

## 4. Explanation System
Provides:
- Why a conclusion was reached
- How rules were triggered
- Transparent AI reasoning

## 5. User Interface
Built using Streamlit to provide:
- Interactive forms
- Stress analysis dashboard
- Visual charts and results

---

# Installation Guide

## 1. Clone Repository

```bash
git clone https://github.com/waqarimran342/StressSense_AI.git
```

---

## 2. Move into Project Directory

```bash
cd StressSense_AI
```

---

## 3. Create Virtual Environment

```bash
python -m venv stress_env
```

---

## 4. Activate Virtual Environment

### Windows

```bash
stress_env\Scripts\activate
```

### Mac/Linux

```bash
source stress_env/bin/activate
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Run the Streamlit application:

```bash
streamlit run interface/app.py
```

The application will open automatically in your browser.

---

# Example Workflow

1. User enters stress-related symptoms
2. Facts are generated
3. Rules are evaluated
4. Inference engine derives conclusions
5. Certainty factors calculate confidence
6. Stress level and recommendations are displayed
7. Explanation system provides reasoning

---

# Example Symptoms Considered

- Anxiety
- Sleep disturbance
- Fatigue
- Overthinking
- Mood swings
- Lack of concentration
- Social withdrawal
- Irritability

---

# AI Concepts Implemented

- Expert Systems
- Knowledge Representation
- Rule-Based Reasoning
- Forward Chaining
- Certainty Factors
- Explainable AI (XAI)

---

# Future Improvements

- Machine Learning Integration
- Personalized Recommendations
- Chatbot Support
- Real-Time Analytics
- Database Integration
- User Authentication
- Mobile Application Support

---

# Educational Purpose

This project was developed as an academic Artificial Intelligence semester project to demonstrate practical implementation of expert systems and knowledge-based reasoning.

---

# Author

Waqar Imran

---

# License

This project is licensed under the MIT License.

---

# GitHub Repository

https://github.com/waqarimran342/StressSense_AI

---

# Tagline

"Smart stress analysis through explainable AI."
