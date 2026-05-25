"""
StressSense AI - Explanation Generator
Provides WHY and HOW explanations for expert system reasoning
"""

from typing import Dict, List, Any, Optional, Tuple
import json


class StressExplainer:
    """
    Generates human-readable explanations for the inference engine's reasoning.
    Supports WHY (justification) and HOW (derivation) explanation modes.
    """

    def __init__(self):
        self.explanation_log: List[Dict] = []
        self.rule_traces: List[Dict] = []
        self.fact_history: List[Dict] = []

        # Human-readable descriptions for symptoms and factors
        self.symptom_labels = {
            # Physical symptoms
            "headaches": "Frequent Headaches",
            "muscle_tension": "Muscle Tension / Tightness",
            "fatigue": "Physical Fatigue",
            "sleep_problems": "Sleep Disturbances",
            "digestive_issues": "Digestive Problems",
            "heart_racing": "Racing Heart / Palpitations",
            "shortness_of_breath": "Shortness of Breath",
            "sweating": "Excessive Sweating",
            "trembling": "Trembling or Shaking",
            "dizziness": "Dizziness or Lightheadedness",
            # Psychological symptoms
            "anxiety_worry": "Persistent Anxiety / Worry",
            "mood_swings": "Frequent Mood Swings",
            "irritability": "Increased Irritability",
            "depression_feelings": "Feelings of Depression",
            "overwhelmed": "Feeling Overwhelmed",
            "cognitive_difficulties": "Cognitive Difficulties (Focus/Memory)",
            "negative_thinking": "Negative Thought Patterns",
            "hopelessness": "Sense of Hopelessness",
            "panic_attacks": "Panic Attacks",
            "social_withdrawal": "Social Withdrawal",
            # Behavioral symptoms
            "appetite_changes": "Changes in Appetite",
            "procrastination": "Increased Procrastination",
            "substance_use": "Increased Substance Use",
            "poor_time_management": "Poor Time Management",
            "neglecting_responsibilities": "Neglecting Responsibilities",
            "increased_errors": "Making More Mistakes",
            "reduced_productivity": "Reduced Productivity",
            "avoidance": "Avoidance Behaviors",
        }

        self.stressor_labels = {
            "work_overload": "Work Overload",
            "relationship_conflict": "Relationship Conflicts",
            "financial_stress": "Financial Stress",
            "health_concerns": "Health Concerns",
            "major_life_changes": "Major Life Changes",
            "academic_pressure": "Academic Pressure",
            "social_isolation": "Social Isolation",
            "trauma_history": "History of Trauma",
            "chronic_illness": "Chronic Illness",
            "caregiving_burden": "Caregiving Responsibilities",
        }

        self.protective_labels = {
            "social_support": "Strong Social Support",
            "regular_exercise": "Regular Exercise",
            "healthy_diet": "Healthy Diet",
            "adequate_sleep": "Adequate Sleep",
            "mindfulness_practice": "Mindfulness / Meditation Practice",
            "hobbies_interests": "Active Hobbies & Interests",
            "professional_help": "Access to Professional Help",
            "work_life_balance": "Good Work-Life Balance",
            "relaxation_techniques": "Relaxation Techniques",
            "positive_outlook": "Positive Outlook / Resilience",
        }

        # Rule explanations mapped to their IDs
        self.rule_explanations = self._build_rule_explanations()

    def _build_rule_explanations(self) -> Dict[str, Dict]:
        """Build human-readable explanations for each rule"""
        return {
            # Physical stress rules
            "R001": {
                "name": "Multiple Physical Symptoms → Physical Stress Response",
                "rationale": "The presence of multiple physical symptoms (headaches, muscle tension, fatigue, digestive issues) together indicates the body is experiencing a significant stress response. The autonomic nervous system activates the 'fight-or-flight' response, causing these somatic manifestations.",
                "evidence_type": "physical",
                "confidence_basis": "Clinical research shows that clusters of somatic complaints are reliable indicators of stress-related physiological activation."
            },
            "R002": {
                "name": "Sleep Disruption → Stress Amplification",
                "rationale": "Poor sleep is both a symptom and amplifier of stress. When sleep is disrupted, the body cannot recover from daily stressors, creating a vicious cycle that elevates overall stress levels.",
                "evidence_type": "physical",
                "confidence_basis": "Sleep deprivation research demonstrates bidirectional relationship with stress hormones (cortisol elevation)."
            },
            "R003": {
                "name": "Cardiovascular Symptoms → Acute Stress Indicator",
                "rationale": "Racing heart and shortness of breath are direct manifestations of sympathetic nervous system activation — the body's emergency stress response. These symptoms suggest the stress response is significantly elevated.",
                "evidence_type": "physical",
                "confidence_basis": "Cardiovascular reactivity is a well-documented stress biomarker in psychophysiological research."
            },
            # Psychological stress rules
            "R010": {
                "name": "Anxiety + Overwhelm → Psychological Stress",
                "rationale": "The combination of persistent worry and feeling overwhelmed indicates the cognitive appraisal system is evaluating demands as exceeding available coping resources — the core definition of psychological stress.",
                "evidence_type": "psychological",
                "confidence_basis": "Lazarus & Folkman's transactional model of stress defines stress as a perceived imbalance between demands and resources."
            },
            "R011": {
                "name": "Mood Instability → Emotional Dysregulation",
                "rationale": "Frequent mood swings and irritability suggest reduced emotional regulation capacity, often caused by chronic stress depleting prefrontal cortex executive functions.",
                "evidence_type": "psychological",
                "confidence_basis": "Neuroscience research shows chronic stress impairs prefrontal cortex functioning, reducing emotional control."
            },
            "R012": {
                "name": "Negative Cognition Pattern → Cognitive Stress Load",
                "rationale": "Persistent negative thinking and hopelessness reflect cognitive distortions that both result from and contribute to stress, creating self-reinforcing patterns of psychological burden.",
                "evidence_type": "psychological",
                "confidence_basis": "Cognitive-behavioral theory identifies rumination and catastrophizing as key stress-maintaining cognitive patterns."
            },
            "R013": {
                "name": "Panic Symptoms → Severe Anxiety Response",
                "rationale": "Panic attacks represent an extreme activation of the threat-response system, indicating severe psychological distress that requires immediate attention.",
                "evidence_type": "psychological",
                "confidence_basis": "DSM-5 criteria and clinical research categorize panic as a severe manifestation of anxiety disorders."
            },
            # Behavioral stress rules
            "R020": {
                "name": "Behavioral Changes → Functional Impairment",
                "rationale": "Changes in behavior such as procrastination, social withdrawal, and neglecting responsibilities indicate that stress has begun to impair daily functioning — a key threshold in stress severity assessment.",
                "evidence_type": "behavioral",
                "confidence_basis": "Functional impairment is a diagnostic criterion for stress-related disorders in clinical psychology."
            },
            "R021": {
                "name": "Substance Use Increase → Maladaptive Coping",
                "rationale": "Increased use of alcohol, caffeine, or other substances as a response to stress indicates maladaptive coping strategies that temporarily reduce distress but worsen long-term stress resilience.",
                "evidence_type": "behavioral",
                "confidence_basis": "Tension reduction theory and research on stress-substance use relationships support this inference."
            },
            # Stressor rules
            "R030": {
                "name": "Work Overload → Occupational Stress",
                "rationale": "Excessive workload with insufficient recovery time is one of the most well-documented sources of chronic stress, linked to burnout and physical health consequences.",
                "evidence_type": "environmental",
                "confidence_basis": "Occupational stress research (Karasek's demand-control model) identifies workload as primary stressor."
            },
            "R031": {
                "name": "Relationship Conflict → Social Stress",
                "rationale": "Interpersonal conflicts are major psychosocial stressors that activate threat-response systems, as human social connection is fundamental to psychological safety.",
                "evidence_type": "environmental",
                "confidence_basis": "Social baseline theory and attachment research highlight relationship quality as a core stress buffer or stressor."
            },
            "R032": {
                "name": "Financial Pressure → Chronic Stress Load",
                "rationale": "Financial insecurity creates persistent, uncontrollable stressors that are particularly damaging because they lack clear resolution timelines and threaten basic security needs.",
                "evidence_type": "environmental",
                "confidence_basis": "Economic stress research shows financial strain is among the strongest predictors of mental health outcomes."
            },
            # Protective factor rules
            "R040": {
                "name": "Social Support → Stress Buffering",
                "rationale": "Strong social connections act as a direct buffer against stress by providing emotional support, practical assistance, and a sense of belonging that moderates stress responses.",
                "evidence_type": "protective",
                "confidence_basis": "Social support buffering hypothesis is one of the most replicated findings in health psychology."
            },
            "R041": {
                "name": "Regular Exercise → Neurobiological Stress Reduction",
                "rationale": "Physical exercise reduces stress through multiple mechanisms: lowering cortisol, increasing endorphins, promoting neuroplasticity, and improving sleep quality.",
                "evidence_type": "protective",
                "confidence_basis": "Exercise neuroscience research shows robust stress-reducing effects across multiple biological pathways."
            },
            "R042": {
                "name": "Mindfulness Practice → Stress Regulation",
                "rationale": "Regular mindfulness practice develops metacognitive awareness and emotional regulation skills that reduce reactivity to stressors and enhance recovery.",
                "evidence_type": "protective",
                "confidence_basis": "MBSR research by Kabat-Zinn and subsequent studies demonstrate significant stress reduction effects."
            },
            # Severity assessment rules
            "R050": {
                "name": "Multiple Domain Impairment → High Stress Level",
                "rationale": "When stress symptoms appear across physical, psychological, and behavioral domains simultaneously, it indicates a systemic stress response that has exceeded adaptive coping capacity.",
                "evidence_type": "synthesis",
                "confidence_basis": "Multi-domain assessment is the gold standard in clinical stress evaluation (PSS, DASS-21 frameworks)."
            },
            "R051": {
                "name": "Duration Factor → Chronic Stress Classification",
                "rationale": "Stress persisting beyond 4-6 weeks transitions from acute to chronic stress, which carries significantly greater health risks including immune suppression and cardiovascular strain.",
                "evidence_type": "synthesis",
                "confidence_basis": "Stress duration research distinguishes acute (adaptive) from chronic (pathological) stress profiles."
            },
            "R052": {
                "name": "Low Protective Factors → Vulnerability Assessment",
                "rationale": "Absence of protective factors (exercise, support, coping skills) makes the system more vulnerable to stressors, effectively amplifying their impact.",
                "evidence_type": "synthesis",
                "confidence_basis": "Stress vulnerability models (diathesis-stress) emphasize protective factor deficits as risk amplifiers."
            },
        }

    def log_inference_step(
        self,
        rule_id: str,
        rule_name: str,
        conditions_met: List[str],
        conclusion: str,
        certainty_factor: float,
        fuzzy_values: Optional[Dict] = None
    ):
        """Log an inference step for later explanation"""
        step = {
            "rule_id": rule_id,
            "rule_name": rule_name,
            "conditions_met": conditions_met,
            "conclusion": conclusion,
            "certainty_factor": certainty_factor,
            "fuzzy_values": fuzzy_values or {},
            "step_number": len(self.rule_traces) + 1
        }
        self.rule_traces.append(step)

    def log_fact(self, fact_name: str, value: Any, source: str = "user_input"):
        """Log a fact for explanation purposes"""
        self.fact_history.append({
            "fact": fact_name,
            "value": value,
            "source": source
        })

    def clear_logs(self):
        """Clear all logs for a new assessment"""
        self.explanation_log.clear()
        self.rule_traces.clear()
        self.fact_history.clear()

    def generate_why_explanation(
        self,
        assessment_result: Dict,
        user_inputs: Dict
    ) -> Dict[str, Any]:
        """
        Generate WHY explanation: Why did the system reach this conclusion?
        Returns structured explanation with evidence chain.
        """
        stress_level = assessment_result.get("stress_level", "unknown")
        overall_cf = assessment_result.get("overall_cf", 0.0)
        fuzzy_score = assessment_result.get("fuzzy_score", 0.0)
        fired_rules = assessment_result.get("fired_rules", [])
        domain_scores = assessment_result.get("domain_scores", {})

        # Build evidence summary
        physical_evidence = self._extract_physical_evidence(user_inputs)
        psychological_evidence = self._extract_psychological_evidence(user_inputs)
        behavioral_evidence = self._extract_behavioral_evidence(user_inputs)
        stressor_evidence = self._extract_stressor_evidence(user_inputs)
        protective_evidence = self._extract_protective_evidence(user_inputs)

        # Build primary reasoning chain
        reasoning_chain = self._build_reasoning_chain(
            stress_level, overall_cf, fuzzy_score,
            domain_scores, fired_rules
        )

        # Generate narrative explanation
        narrative = self._generate_why_narrative(
            stress_level, overall_cf,
            physical_evidence, psychological_evidence,
            behavioral_evidence, stressor_evidence, protective_evidence
        )

        return {
            "type": "WHY",
            "conclusion": {
                "stress_level": stress_level,
                "certainty_factor": overall_cf,
                "fuzzy_score": fuzzy_score,
                "confidence_label": self._cf_to_label(overall_cf)
            },
            "evidence": {
                "physical": physical_evidence,
                "psychological": psychological_evidence,
                "behavioral": behavioral_evidence,
                "stressors": stressor_evidence,
                "protective_factors": protective_evidence
            },
            "reasoning_chain": reasoning_chain,
            "narrative": narrative,
            "domain_scores": domain_scores,
            "rules_fired_count": len(fired_rules),
            "key_contributing_factors": self._identify_key_factors(
                user_inputs, domain_scores
            )
        }

    def generate_how_explanation(
        self,
        assessment_result: Dict,
        user_inputs: Dict
    ) -> Dict[str, Any]:
        """
        Generate HOW explanation: How did the system derive this conclusion step by step?
        Returns detailed step-by-step derivation.
        """
        fired_rules = assessment_result.get("fired_rules", [])
        domain_scores = assessment_result.get("domain_scores", {})

        # Build step-by-step derivation
        derivation_steps = self._build_derivation_steps(fired_rules, user_inputs)

        # Build fuzzy logic explanation
        fuzzy_explanation = self._explain_fuzzy_logic(assessment_result)

        # Build CF combination explanation
        cf_explanation = self._explain_cf_combination(assessment_result)

        # Build domain synthesis explanation
        domain_explanation = self._explain_domain_synthesis(domain_scores)

        return {
            "type": "HOW",
            "total_steps": len(derivation_steps),
            "derivation_steps": derivation_steps,
            "fuzzy_logic_explanation": fuzzy_explanation,
            "certainty_factor_explanation": cf_explanation,
            "domain_synthesis": domain_explanation,
            "inference_method": {
                "name": "Forward Chaining with Fuzzy Logic and Certainty Factors",
                "description": (
                    "The system uses forward chaining to match user-provided facts "
                    "against condition patterns. Fuzzy membership functions convert "
                    "symptom counts to degrees of truth. Certainty factors combine "
                    "rule confidences using the CF algebra to produce a final assessment."
                ),
                "steps_summary": [
                    "1. User inputs collected and converted to working memory facts",
                    "2. Forward chaining matches facts to rule antecedents",
                    "3. Fuzzy logic computes membership degrees for symptom clusters",
                    "4. Certainty factors propagated through rule firings",
                    "5. Domain scores aggregated with weighted combination",
                    "6. Final stress level determined by CF + fuzzy composite score"
                ]
            }
        }

    def generate_confidence_explanation(self, cf: float, rule_id: str = None) -> str:
        """Explain the confidence level of a specific conclusion"""
        label = self._cf_to_label(cf)
        pct = int(cf * 100)

        base = f"Confidence: {pct}% ({label}). "

        if cf >= 0.85:
            detail = (
                "This high confidence is supported by strong, consistent evidence "
                "across multiple symptom domains with high certainty factor values."
            )
        elif cf >= 0.65:
            detail = (
                "This moderate-high confidence reflects clear evidence in several "
                "domains, though some uncertainty remains due to missing or "
                "ambiguous symptom information."
            )
        elif cf >= 0.45:
            detail = (
                "This moderate confidence indicates the evidence points in this "
                "direction but is not conclusive — some symptoms may be absent "
                "or conflicting evidence was found."
            )
        elif cf >= 0.25:
            detail = (
                "This low-moderate confidence reflects limited or weak evidence. "
                "The conclusion is tentative and additional information would "
                "significantly change the assessment."
            )
        else:
            detail = (
                "This low confidence indicates minimal evidence for this conclusion. "
                "The assessment is highly uncertain."
            )

        if rule_id and rule_id in self.rule_explanations:
            rule_info = self.rule_explanations[rule_id]
            detail += f" The primary rule ({rule_info['name']}) contributes based on: {rule_info['confidence_basis']}"

        return base + detail

    def generate_recommendation_explanation(
        self,
        stress_level: str,
        recommendations: List[str],
        user_inputs: Dict
    ) -> Dict[str, Any]:
        """Generate explanations for why specific recommendations are made"""
        explained_recommendations = []

        for rec in recommendations:
            explanation = self._explain_recommendation(rec, stress_level, user_inputs)
            explained_recommendations.append({
                "recommendation": rec,
                "rationale": explanation["rationale"],
                "evidence_basis": explanation["evidence_basis"],
                "priority": explanation["priority"]
            })

        # Sort by priority
        explained_recommendations.sort(key=lambda x: x["priority"])

        return {
            "stress_level": stress_level,
            "recommendations_count": len(explained_recommendations),
            "recommendations": explained_recommendations,
            "general_rationale": self._general_recommendation_rationale(stress_level)
        }

    def format_explanation_for_display(
        self,
        explanation: Dict,
        mode: str = "WHY"
    ) -> List[Dict]:
        """
        Format explanation into display-ready sections for the UI.
        Returns a list of sections with type, title, and content.
        """
        sections = []

        if mode == "WHY":
            sections = self._format_why_for_display(explanation)
        elif mode == "HOW":
            sections = self._format_how_for_display(explanation)

        return sections

    # ─── Private Helper Methods ───────────────────────────────────────────────

    def _extract_physical_evidence(self, user_inputs: Dict) -> List[Dict]:
        physical_symptoms = [
            "headaches", "muscle_tension", "fatigue", "sleep_problems",
            "digestive_issues", "heart_racing", "shortness_of_breath",
            "sweating", "trembling", "dizziness"
        ]
        evidence = []
        for sym in physical_symptoms:
            val = user_inputs.get(sym, False)
            if val:
                evidence.append({
                    "symptom": sym,
                    "label": self.symptom_labels.get(sym, sym.replace("_", " ").title()),
                    "present": True,
                    "severity": user_inputs.get(f"{sym}_severity", "moderate")
                })
        return evidence

    def _extract_psychological_evidence(self, user_inputs: Dict) -> List[Dict]:
        psych_symptoms = [
            "anxiety_worry", "mood_swings", "irritability", "depression_feelings",
            "overwhelmed", "cognitive_difficulties", "negative_thinking",
            "hopelessness", "panic_attacks", "social_withdrawal"
        ]
        evidence = []
        for sym in psych_symptoms:
            val = user_inputs.get(sym, False)
            if val:
                evidence.append({
                    "symptom": sym,
                    "label": self.symptom_labels.get(sym, sym.replace("_", " ").title()),
                    "present": True,
                    "severity": user_inputs.get(f"{sym}_severity", "moderate")
                })
        return evidence

    def _extract_behavioral_evidence(self, user_inputs: Dict) -> List[Dict]:
        behavioral_symptoms = [
            "appetite_changes", "procrastination", "substance_use",
            "poor_time_management", "neglecting_responsibilities",
            "increased_errors", "reduced_productivity", "avoidance"
        ]
        evidence = []
        for sym in behavioral_symptoms:
            val = user_inputs.get(sym, False)
            if val:
                evidence.append({
                    "symptom": sym,
                    "label": self.symptom_labels.get(sym, sym.replace("_", " ").title()),
                    "present": True
                })
        return evidence

    def _extract_stressor_evidence(self, user_inputs: Dict) -> List[Dict]:
        stressors = list(self.stressor_labels.keys())
        evidence = []
        for stressor in stressors:
            val = user_inputs.get(stressor, False)
            if val:
                evidence.append({
                    "stressor": stressor,
                    "label": self.stressor_labels.get(stressor, stressor.replace("_", " ").title()),
                    "present": True,
                    "intensity": user_inputs.get(f"{stressor}_intensity", "moderate")
                })
        return evidence

    def _extract_protective_evidence(self, user_inputs: Dict) -> List[Dict]:
        factors = list(self.protective_labels.keys())
        present = []
        absent = []
        for factor in factors:
            val = user_inputs.get(factor, False)
            entry = {
                "factor": factor,
                "label": self.protective_labels.get(factor, factor.replace("_", " ").title()),
                "present": bool(val)
            }
            if val:
                present.append(entry)
            else:
                absent.append(entry)
        return {"present": present, "absent": absent}

    def _build_reasoning_chain(
        self,
        stress_level: str,
        overall_cf: float,
        fuzzy_score: float,
        domain_scores: Dict,
        fired_rules: List
    ) -> List[Dict]:
        chain = []

        # Step 1: Data collection
        chain.append({
            "step": 1,
            "type": "data_collection",
            "title": "Evidence Collection",
            "description": "User-reported symptoms, stressors, and protective factors were collected and converted into working memory facts."
        })

        # Step 2: Domain analysis
        for domain, score in domain_scores.items():
            if score > 0.1:
                chain.append({
                    "step": len(chain) + 1,
                    "type": "domain_analysis",
                    "title": f"{domain.replace('_', ' ').title()} Domain Analysis",
                    "description": f"Domain score of {score:.2f} computed from relevant symptom cluster.",
                    "score": score
                })

        # Step 3: Rule firing summary
        chain.append({
            "step": len(chain) + 1,
            "type": "rule_application",
            "title": "Rule Application",
            "description": f"{len(fired_rules)} inference rules fired based on the evidence.",
            "rules_count": len(fired_rules)
        })

        # Step 4: CF combination
        chain.append({
            "step": len(chain) + 1,
            "type": "cf_combination",
            "title": "Certainty Factor Combination",
            "description": f"Individual rule certainty factors combined to yield overall CF of {overall_cf:.2f}.",
            "cf": overall_cf
        })

        # Step 5: Fuzzy aggregation
        chain.append({
            "step": len(chain) + 1,
            "type": "fuzzy_aggregation",
            "title": "Fuzzy Score Aggregation",
            "description": f"Fuzzy membership functions produced composite score of {fuzzy_score:.2f}.",
            "fuzzy_score": fuzzy_score
        })

        # Step 6: Final conclusion
        chain.append({
            "step": len(chain) + 1,
            "type": "conclusion",
            "title": "Final Assessment",
            "description": f"Stress level classified as '{stress_level}' based on composite score and certainty factors.",
            "conclusion": stress_level,
            "cf": overall_cf
        })

        return chain

    def _build_derivation_steps(
        self,
        fired_rules: List,
        user_inputs: Dict
    ) -> List[Dict]:
        steps = []

        # Step 0: Initial facts
        active_symptoms = []
        for key, val in user_inputs.items():
            if val and isinstance(val, bool):
                label = self.symptom_labels.get(
                    key,
                    self.stressor_labels.get(
                        key,
                        self.protective_labels.get(key, key.replace("_", " ").title())
                    )
                )
                active_symptoms.append(label)

        steps.append({
            "step_number": 0,
            "type": "initial_facts",
            "title": "Initial Working Memory",
            "description": "The following facts were established from user input:",
            "facts": active_symptoms[:15],  # limit display
            "facts_total": len(active_symptoms)
        })

        # Rules steps
        for i, rule in enumerate(fired_rules[:12]):  # Show up to 12 rules
            rule_id = rule.get("rule_id", f"R{i:03d}")
            rule_info = self.rule_explanations.get(rule_id, {})
            conditions = rule.get("conditions", [])
            conclusion = rule.get("conclusion", "Unknown conclusion")
            cf = rule.get("cf", 0.5)

            steps.append({
                "step_number": i + 1,
                "type": "rule_firing",
                "rule_id": rule_id,
                "title": rule_info.get("name", rule.get("name", f"Rule {rule_id}")),
                "conditions": conditions,
                "conclusion": conclusion,
                "cf": cf,
                "rationale": rule_info.get("rationale", ""),
                "evidence_type": rule_info.get("evidence_type", "general"),
                "confidence_basis": rule_info.get("confidence_basis", "")
            })

        return steps

    def _explain_fuzzy_logic(self, assessment_result: Dict) -> Dict:
        fuzzy_score = assessment_result.get("fuzzy_score", 0.0)
        domain_scores = assessment_result.get("domain_scores", {})

        memberships = {}
        for level, (low, high) in {
            "minimal": (0.0, 0.25),
            "mild": (0.15, 0.45),
            "moderate": (0.35, 0.65),
            "high": (0.55, 0.80),
            "severe": (0.70, 1.0)
        }.items():
            if fuzzy_score <= low:
                memberships[level] = 0.0
            elif fuzzy_score >= high:
                memberships[level] = 1.0
            else:
                memberships[level] = (fuzzy_score - low) / (high - low)

        return {
            "composite_score": fuzzy_score,
            "interpretation": (
                "Fuzzy logic converts the crisp symptom counts and domain scores "
                "into degrees of membership in stress level categories. Unlike binary "
                "logic, fuzzy membership allows a score to partially belong to "
                "multiple categories."
            ),
            "membership_degrees": memberships,
            "dominant_category": max(memberships, key=memberships.get),
            "domain_contributions": {
                k: round(v, 3) for k, v in domain_scores.items()
            }
        }

    def _explain_cf_combination(self, assessment_result: Dict) -> Dict:
        overall_cf = assessment_result.get("overall_cf", 0.0)
        fired_rules = assessment_result.get("fired_rules", [])

        individual_cfs = [r.get("cf", 0.5) for r in fired_rules if r.get("cf")]
        avg_cf = sum(individual_cfs) / len(individual_cfs) if individual_cfs else 0.0
        max_cf = max(individual_cfs) if individual_cfs else 0.0

        return {
            "final_cf": overall_cf,
            "rules_contributing": len(individual_cfs),
            "average_rule_cf": round(avg_cf, 3),
            "highest_rule_cf": round(max_cf, 3),
            "combination_formula": (
                "CF(A,B) = CF(A) + CF(B) × (1 - CF(A))  when both positive\n"
                "CF(A,B) = CF(A) + CF(B) × (1 + CF(A))  when both negative\n"
                "CF(A,B) = (CF(A) + CF(B)) / (1 - min(|CF(A)|, |CF(B)|))  when mixed"
            ),
            "interpretation": (
                "Certainty Factors accumulate evidence across multiple rules. "
                f"With {len(individual_cfs)} rules contributing, the combined "
                f"confidence of {overall_cf:.1%} reflects the cumulative evidence weight."
            )
        }

    def _explain_domain_synthesis(self, domain_scores: Dict) -> Dict:
        weights = {
            "physical": 0.25,
            "psychological": 0.35,
            "behavioral": 0.20,
            "stressors": 0.15,
            "protective": -0.05
        }

        contributions = {}
        for domain, score in domain_scores.items():
            weight = weights.get(domain, 0.1)
            contributions[domain] = {
                "raw_score": round(score, 3),
                "weight": weight,
                "weighted_contribution": round(score * abs(weight), 3),
                "direction": "increases" if weight > 0 else "decreases"
            }

        return {
            "domain_weights": weights,
            "contributions": contributions,
            "synthesis_method": (
                "Domain scores are computed independently then combined with "
                "clinical weights. Psychological symptoms carry the highest weight "
                "as they most directly reflect subjective stress experience. "
                "Physical symptoms are weighted second as objective stress indicators. "
                "Protective factors reduce the final score."
            )
        }

    def _generate_why_narrative(
        self,
        stress_level: str,
        overall_cf: float,
        physical: List,
        psychological: List,
        behavioral: List,
        stressors: List,
        protective: Dict
    ) -> str:
        pct = int(overall_cf * 100)
        parts = []

        parts.append(
            f"The system assessed your stress level as **{stress_level.upper()}** "
            f"with {pct}% confidence based on the following reasoning:"
        )

        if physical:
            sym_list = ", ".join(e["label"] for e in physical[:3])
            extra = f" and {len(physical) - 3} more" if len(physical) > 3 else ""
            parts.append(
                f"• **Physical Evidence** ({len(physical)} symptoms): "
                f"{sym_list}{extra} were reported. These physical manifestations "
                "indicate your body's stress-response system is activated."
            )

        if psychological:
            sym_list = ", ".join(e["label"] for e in psychological[:3])
            extra = f" and {len(psychological) - 3} more" if len(psychological) > 3 else ""
            parts.append(
                f"• **Psychological Evidence** ({len(psychological)} symptoms): "
                f"{sym_list}{extra} were identified, pointing to cognitive "
                "and emotional stress burden."
            )

        if behavioral:
            sym_list = ", ".join(e["label"] for e in behavioral[:2])
            parts.append(
                f"• **Behavioral Evidence** ({len(behavioral)} changes): "
                f"{sym_list} and related changes suggest stress is affecting "
                "your day-to-day functioning."
            )

        if stressors:
            s_list = ", ".join(e["label"] for e in stressors[:3])
            parts.append(
                f"• **Active Stressors** ({len(stressors)} identified): "
                f"{s_list} are contributing to your current stress load."
            )

        pf_present = protective.get("present", [])
        pf_absent = protective.get("absent", [])
        if pf_present:
            pf_list = ", ".join(e["label"] for e in pf_present[:2])
            parts.append(
                f"• **Protective Factors** ({len(pf_present)} active): "
                f"{pf_list} are helping to moderate your stress levels."
            )
        if len(pf_absent) > len(pf_present):
            parts.append(
                f"• **Vulnerability** ({len(pf_absent)} protective factors absent): "
                "The limited number of active coping resources increases vulnerability."
            )

        return "\n\n".join(parts)

    def _identify_key_factors(
        self,
        user_inputs: Dict,
        domain_scores: Dict
    ) -> List[Dict]:
        factors = []

        # Find highest domain
        if domain_scores:
            top_domain = max(domain_scores, key=domain_scores.get)
            factors.append({
                "factor": top_domain.replace("_", " ").title(),
                "impact": "highest",
                "score": domain_scores[top_domain],
                "description": f"This domain showed the strongest evidence for stress."
            })

        # Check for severe symptoms
        severe_symptoms = ["panic_attacks", "hopelessness", "heart_racing", "shortness_of_breath"]
        for sym in severe_symptoms:
            if user_inputs.get(sym):
                factors.append({
                    "factor": self.symptom_labels.get(sym, sym),
                    "impact": "high",
                    "description": "This symptom is a strong clinical indicator of significant stress."
                })

        # Check for missing protective factors
        key_protectives = ["social_support", "regular_exercise", "adequate_sleep"]
        missing = [p for p in key_protectives if not user_inputs.get(p)]
        if missing:
            labels = [self.protective_labels.get(p, p) for p in missing]
            factors.append({
                "factor": "Missing Key Coping Resources",
                "impact": "amplifying",
                "description": f"Absence of: {', '.join(labels)} amplifies stress vulnerability."
            })

        return factors[:5]

    def _cf_to_label(self, cf: float) -> str:
        if cf >= 0.85:
            return "Very High Confidence"
        elif cf >= 0.65:
            return "High Confidence"
        elif cf >= 0.45:
            return "Moderate Confidence"
        elif cf >= 0.25:
            return "Low-Moderate Confidence"
        else:
            return "Low Confidence"

    def _explain_recommendation(
        self,
        recommendation: str,
        stress_level: str,
        user_inputs: Dict
    ) -> Dict:
        rec_lower = recommendation.lower()
        priority = 3  # default

        if any(word in rec_lower for word in ["crisis", "emergency", "immediate", "professional"]):
            priority = 1
            rationale = (
                "Given the severity of your stress assessment, professional support "
                "is the most important step. A qualified mental health professional "
                "can provide personalized treatment that addresses the root causes."
            )
            evidence_basis = "Clinical guidelines recommend professional intervention for high/severe stress."
        elif any(word in rec_lower for word in ["sleep", "rest", "exercise", "physical"]):
            priority = 2
            rationale = (
                "Physical health interventions directly address the physiological "
                "stress response by reducing cortisol levels and promoting recovery."
            )
            evidence_basis = "Exercise and sleep hygiene have strong evidence bases for stress reduction."
        elif any(word in rec_lower for word in ["mindful", "meditat", "breath", "relax"]):
            priority = 2
            rationale = (
                "Mindfulness and relaxation techniques activate the parasympathetic "
                "nervous system, counteracting the stress response."
            )
            evidence_basis = "MBSR and relaxation response research (Benson) demonstrate reliable effects."
        elif any(word in rec_lower for word in ["social", "connect", "talk", "support"]):
            priority = 2
            rationale = (
                "Social connection is one of the most powerful stress buffers. "
                "Sharing problems with trusted others reduces subjective stress burden."
            )
            evidence_basis = "Social support buffering is among the most replicated findings in health psychology."
        else:
            priority = 3
            rationale = (
                "This recommendation targets lifestyle factors that build long-term "
                "stress resilience and reduce overall vulnerability."
            )
            evidence_basis = "General well-being research supports lifestyle-based stress management."

        return {
            "rationale": rationale,
            "evidence_basis": evidence_basis,
            "priority": priority
        }

    def _general_recommendation_rationale(self, stress_level: str) -> str:
        rationale_map = {
            "minimal": (
                "Your stress is well-managed. These recommendations focus on "
                "maintaining your current resilience and preventing future stress escalation."
            ),
            "mild": (
                "Your stress is manageable but worth addressing proactively. "
                "Early intervention prevents mild stress from becoming chronic."
            ),
            "moderate": (
                "Your stress level warrants active management. These recommendations "
                "target the most impactful intervention points identified in your assessment."
            ),
            "high": (
                "Your stress level is significantly elevated. These recommendations "
                "prioritize immediate relief strategies alongside longer-term interventions."
            ),
            "severe": (
                "Your stress level is in a range that carries health risks. "
                "Professional support is strongly recommended alongside self-care strategies."
            )
        }
        return rationale_map.get(
            stress_level.lower(),
            "These recommendations are tailored to your specific stress profile."
        )

    def _format_why_for_display(self, explanation: Dict) -> List[Dict]:
        sections = []

        # Conclusion section
        conclusion = explanation.get("conclusion", {})
        sections.append({
            "type": "conclusion_summary",
            "title": "Assessment Conclusion",
            "stress_level": conclusion.get("stress_level", "Unknown"),
            "certainty_factor": conclusion.get("certainty_factor", 0.0),
            "fuzzy_score": conclusion.get("fuzzy_score", 0.0),
            "confidence_label": conclusion.get("confidence_label", ""),
        })

        # Narrative
        sections.append({
            "type": "narrative",
            "title": "Why This Assessment?",
            "content": explanation.get("narrative", "")
        })

        # Evidence breakdown
        evidence = explanation.get("evidence", {})
        physical = evidence.get("physical", [])
        psychological = evidence.get("psychological", [])
        behavioral = evidence.get("behavioral", [])
        stressors = evidence.get("stressors", [])
        protective = evidence.get("protective_factors", {})

        sections.append({
            "type": "evidence_grid",
            "title": "Evidence Summary",
            "physical_count": len(physical),
            "psychological_count": len(psychological),
            "behavioral_count": len(behavioral),
            "stressors_count": len(stressors),
            "protective_present": len(protective.get("present", [])),
            "protective_absent": len(protective.get("absent", [])),
            "physical_items": physical,
            "psychological_items": psychological,
            "behavioral_items": behavioral,
            "stressor_items": stressors,
            "protective_items": protective
        })

        # Key factors
        sections.append({
            "type": "key_factors",
            "title": "Key Contributing Factors",
            "factors": explanation.get("key_contributing_factors", [])
        })

        # Reasoning chain
        sections.append({
            "type": "reasoning_chain",
            "title": "Reasoning Chain",
            "steps": explanation.get("reasoning_chain", [])
        })

        return sections

    def _format_how_for_display(self, explanation: Dict) -> List[Dict]:
        sections = []

        # Inference method overview
        method = explanation.get("inference_method", {})
        sections.append({
            "type": "method_overview",
            "title": "Inference Method",
            "name": method.get("name", ""),
            "description": method.get("description", ""),
            "steps_summary": method.get("steps_summary", [])
        })

        # Derivation steps
        sections.append({
            "type": "derivation_steps",
            "title": "Step-by-Step Derivation",
            "total_steps": explanation.get("total_steps", 0),
            "steps": explanation.get("derivation_steps", [])
        })

        # Fuzzy logic
        sections.append({
            "type": "fuzzy_explanation",
            "title": "Fuzzy Logic Analysis",
            "data": explanation.get("fuzzy_logic_explanation", {})
        })

        # CF combination
        sections.append({
            "type": "cf_explanation",
            "title": "Certainty Factor Combination",
            "data": explanation.get("certainty_factor_explanation", {})
        })

        # Domain synthesis
        sections.append({
            "type": "domain_synthesis",
            "title": "Domain Score Synthesis",
            "data": explanation.get("domain_synthesis", {})
        })

        return sections