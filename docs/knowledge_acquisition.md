# Knowledge Acquisition Documentation
## Mental Stress Screening Expert System

---

## 1. Overview

This document describes how expert knowledge was acquired,
structured, and validated for the Mental Stress Screening System.
The system uses 66 production rules derived from published clinical
literature and validated psychological assessment tools.

---

## 2. Knowledge Sources

| Source | Type | Year | Used For |
|--------|------|------|---------|
| DSM-5 (APA) | Clinical Manual | 2013 | Symptom definitions |
| PSS-14 | Validated Scale | 1983 | Symptom weighting |
| Holmes-Rahe Inventory | Validated Scale | 1967 | Trigger scoring |
| GAD-7 | Validated Scale | 2006 | Anxiety rules |
| PHQ-9 | Validated Scale | 2001 | Depression rules |
| Maslach Burnout Inventory | Validated Scale | 1981 | Burnout pattern rules |
| WHO mhGAP | Clinical Guide | 2016 | Recommendations |
| ICD-11 | Classification | 2019 | Stress categories |
| MYCIN System | AI Reference | 1976 | CF methodology |
| Zadeh Fuzzy Sets | AI Reference | 1965 | Fuzzy logic foundation |

---

## 3. Knowledge Elicitation Method
Step 1: Literature review of all sources above
Step 2: Extract all mentioned stress indicators
Step 3: Classify indicators into 5 categories
Step 4: Assign severity levels (none/mild/moderate/severe/frequent)
Step 5: Assign point weights based on clinical significance
Step 6: Assign certainty factors based on diagnostic reliability
Step 7: Assign salience based on urgency priority
Step 8: Design compound rules for known co-occurrence patterns
Step 9: Design protective rules for health-positive behaviours
Step 10: Test and validate all rules against 8 scenarios

text


---

## 4. Rule Categories and Justification

### Category 1: Physical Symptoms (R1-R15)
Physical manifestations are commonly associated with the
autonomic stress response (fight-or-flight activation).
Sources: DSM-5, clinical stress literature.

### Category 2: Psychological Symptoms (R16-R30)
Core psychological indicators from GAD-7, PHQ-9 and DSM-5.
These carry higher certainty factors due to direct diagnostic value.

### Category 3: Behavioural Symptoms (R31-R40)
Behavioural changes are strong indicators of chronic stress
per Holmes-Rahe and Maslach frameworks.

### Category 4: Lifestyle and Triggers (R41-R50)
Life stressors weighted according to Holmes-Rahe stress units.
Lifestyle protective factors based on WHO exercise guidelines.

### Category 5: Compound and Meta Rules (R51-R65)
Co-occurrence of symptoms increases diagnostic confidence.
Compound rules have higher salience than individual symptom rules.

---

## 5. Certainty Factor Assignment

| CF Range | Meaning | Example Symptoms |
|----------|---------|-----------------|
| 0.90 – 1.00 | Very high diagnostic confidence | Self-harm, hopelessness, depression |
| 0.70 – 0.89 | High confidence | Panic attacks, severe anxiety, burnout |
| 0.50 – 0.69 | Moderate confidence | Muscle tension, stomach issues |
| 0.30 – 0.49 | Low confidence | Occasional symptoms |
| Negative CF | Protective factors | Regular exercise, strong social support |

---

## 6. Salience Assignment

| Priority | Salience | Rationale |
|----------|---------|-----------|
| EMERGENCY | 100 | Must always fire first regardless of other rules |
| Critical Compound | 28-35 | Multiple severe indicators present simultaneously |
| High Compound | 20-27 | Two significant indicators together |
| High Single | 15-19 | Highly specific individual symptom |
| Moderate | 8-14 | Moderately specific symptom |
| Low | 1-7 | Non-specific or mild symptom |
| Classification | 1 | Must fire last after all evidence collected |

---

## 7. Validation

All 8 test scenarios were run and results compared to
expected outputs based on clinical reasoning.
Pass rate target: 100% of test cases within expected range.

---

## 8. Limitations

- System is based on self-reported symptoms (no objective measures)
- No longitudinal tracking of symptoms over time
- Cultural factors in stress expression not fully modelled
- System cannot replace clinical interview and professional judgement
- Emergency rules provide guidance only — not a substitute for crisis services