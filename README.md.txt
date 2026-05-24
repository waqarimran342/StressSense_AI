# 🧠 Mental Stress Screening System
### AI-Powered Expert System with Fuzzy Logic & Explainable Reasoning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Experta](https://img.shields.io/badge/Experta-1.9.4-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**An intelligent rule-based expert system for mental stress screening**  
*Built as an Artificial Intelligence semester project*

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-system-architecture) • [Rules](#-knowledge-base) • [Team](#-team)

</div>

---

## ⚠️ Disclaimer

> **This system is a screening tool only and is NOT a substitute for professional medical or psychological diagnosis.**  
> If you or someone you know is experiencing a mental health crisis, please contact a qualified mental health professional immediately.
>
> **Pakistan Crisis Helplines:**
> - Umang Mental Health Helpline: **0317-4288665**
> - Rozan Counseling Center: **051-2890505**  
> - Emergency Services: **115**

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Demo](#-demo)
- [System Architecture](#-system-architecture)
- [Knowledge Base](#-knowledge-base)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [AI Concepts Used](#-ai-concepts-implemented)
- [Testing](#-testing)
- [Screenshots](#-screenshots)
- [Team](#-team)
- [Acknowledgements](#-acknowledgements)

---

## 📖 About the Project

The **Mental Stress Screening System** is a rule-based expert system developed as part of an Artificial Intelligence course project. It simulates the diagnostic reasoning of a mental health professional by using:

- **65+ handcrafted inference rules** organized across 6 categories
- **Forward chaining** inference engine (built with Python Experta)
- **Fuzzy logic** for gradual/uncertain stress level classification
- **Certainty factors (CF)** for managing diagnostic uncertainty
- **Explainable AI** with full WHY and HOW reasoning explanations
- **Conflict resolution** using salience-based rule prioritization
- **Interactive web interface** built with Streamlit

### 🎯 Project Goals

| Goal | Description |
|------|-------------|
| Knowledge Representation | Model expert knowledge about mental stress using structured facts and rules |
| Inference Engine | Implement forward chaining to derive conclusions from patient symptoms |
| Uncertainty Handling | Use certainty factors and fuzzy logic for uncertain diagnoses |
| Explainability | Provide transparent reasoning so users understand every conclusion |
| Usability | Deliver findings through a clean, accessible web interface |

---

## ✨ Features

### Core Expert System Features
- ✅ **65+ Inference Rules** across 6 symptom categories
- ✅ **Forward Chaining** inference engine
- ✅ **Certainty Factor (CF)** combination using standard formula
- ✅ **Conflict Resolution** via salience-based prioritization
- ✅ **Explanation Module** with WHY and HOW reasoning
- ✅ **Knowledge Acquisition Documentation**
- ✅ **Multi-scenario Testing** with automated test suite

### Advanced / Bonus Features
- 🌟 **Fuzzy Logic Classification** with membership functions
- 🌟 **Compound Rules** for multi-symptom pattern detection
- 🌟 **Protective Factor Rules** (negative certainty factors)
- 🌟 **Emergency Detection** with highest-priority rules
- 🌟 **Web Deployment** via Streamlit
- 🌟 **Reasoning Chain Visualization** with Plotly
- 🌟 **Rule Audit Trail** for complete transparency

### Stress Categories Detected
| Category | Description |
|----------|-------------|
| 🟢 Minimal | No significant stress indicators |
| 🟡 Mild | Minor stress, manageable with lifestyle changes |
| 🟠 Moderate | Noticeable stress affecting daily function |
| 🔴 Severe | High stress requiring professional attention |
| ⛔ Critical | Crisis-level stress, immediate intervention needed |

---

## 🎬 Demo

### Live Application
> Run locally using the instructions in the [Installation](#-installation) section.

### Quick Preview