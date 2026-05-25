# System Design Documentation
## Mental Stress Screening Expert System

---

## 1. System Overview

The Mental Stress Screening System is a rule-based expert system
that simulates the structured diagnostic reasoning of a mental health
clinician screening for stress-related conditions.

**Core Technology Stack:**
- Language: Python 3.8+
- Expert System: Experta (Python CLIPS port)
- Fuzzy Logic: Custom implementation with NumPy
- Interface: Streamlit
- Visualisation: Plotly

---

## 2. Architecture

User Input (Streamlit Form)
↓
StressScreeningEngine.run_screening(patient_data)
↓
┌───┴────────────────────────────────────┐
│ Inference Engine │
│ 1. Assert Facts │
│ 2. Forward Chaining (Experta) │
│ 3. Conflict Resolution (Salience) │
│ 4. CF Combination (MYCIN) │
└───┬────────────────────────────────────┘
↓
FuzzyStressSystem.analyze(score)
↓
StressExplainer.generate_full_report()
↓
Return results dict to Streamlit


---

## 3. Inference Strategy

**Type:** Forward Chaining (Data-Driven)

The system starts with patient-reported facts and works forward
to derive conclusions. Rules fire when their conditions match
facts in working memory. This continues until no more rules can fire.

**Why Forward Chaining?**
- Patient data is available upfront (assessment form)
- We want to derive ALL applicable conclusions
- More natural for screening: data → diagnosis
- Easier to explain the reasoning chain

---

## 4. Conflict Resolution

When multiple rules can fire simultaneously, the rule with the
highest salience is selected. This implements priority-based
conflict resolution ensuring:
1. Emergency situations are always addressed first
2. Compound patterns take precedence over individual symptoms
3. Classification happens only after all evidence is gathered

---

## 5. Uncertainty Management

**Certainty Factors (MYCIN method):**
Each rule has an assigned CF (−1.0 to +1.0).
CFs are combined after each rule fires using the MYCIN formula.
Final CF represents overall confidence in the stress diagnosis.

**Fuzzy Logic:**
The crisp stress score is fuzzified to handle the gradual
and overlapping nature of stress levels.
Defuzzification uses the centroid method.

---

## 6. Explanation Facility

Three types of explanations are provided:

| Type | Content |
|------|---------|
| WHY | Top evidence rules and their impact on the conclusion |
| HOW | Step-by-step inference process description |
| AUDIT | Complete log of every rule that fired |

---

## 7. Testing Strategy

8 automated test cases cover:
- Healthy baseline (no stress)
- Mild stress scenarios
- Moderate stress scenarios
- Severe burnout patterns
- Critical depression risk
- Protected individuals (lifestyle factors)
- Multiple physical symptoms
- Academic stress patterns