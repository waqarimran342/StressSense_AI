"""
Inference Engine - Main Orchestrator for StressSense AI
Implements forward-chaining inference with certainty factors
"""

from typing import Dict, List, Optional, Tuple, Any
from knowledge_base.facts import Fact, FactBase, FactType, StressLevel
from knowledge_base.rules import RuleBase, Rule
from inference.fuzzy_logic import FuzzyMembership, FuzzyRuleEvaluator
from inference.certainty_factors import CertaintyFactorCalculator, CFTracker
import time


class InferenceResult:
    """
    Complete result from the inference engine
    Contains all conclusions, evidence, and explanations
    """
    
    def __init__(self):
        self.stress_level: str = "unknown"
        self.stress_score: float = 0.0
        self.certainty_factor: float = 0.0
        self.fuzzy_score: float = 0.0
        
        # Domain assessments
        self.physical_stress_cf: float = 0.0
        self.psychological_stress_cf: float = 0.0
        self.lifestyle_risk_cf: float = 0.0
        self.social_isolation_cf: float = 0.0
        self.burnout_risk_cf: float = 0.0
        self.anxiety_disorder_risk_cf: float = 0.0
        self.poor_sleep_cf: float = 0.0
        
        # Fired rules
        self.fired_rules: List[Dict] = []
        self.fired_rule_ids: List[str] = []
        
        # Derived facts
        self.derived_facts: Dict[str, float] = {}
        
        # Fuzzy memberships
        self.stress_memberships: Dict[str, float] = {}
        
        # Recommendations
        self.recommendations: List[str] = []
        self.professional_help_needed: bool = False
        
        # Metadata
        self.inference_time: float = 0.0
        self.rules_evaluated: int = 0
        self.facts_used: int = 0
        
        # CF Tracker
        self.cf_tracker: CFTracker = CFTracker()
    
    def get_stress_category(self) -> str:
        """Get clean stress category name"""
        mapping = {
            'stress_level_minimal': 'Minimal',
            'stress_level_low': 'Low',
            'stress_level_moderate': 'Moderate',
            'stress_level_high': 'High',
            'stress_level_severe': 'Severe',
            'stress_level_critical': 'Critical'
        }
        return mapping.get(self.stress_level, 'Unknown')
    
    def get_color(self) -> str:
        """Get color code for stress level"""
        colors = {
            'stress_level_minimal': '#27AE60',
            'stress_level_low': '#82E0AA',
            'stress_level_moderate': '#F39C12',
            'stress_level_high': '#E67E22',
            'stress_level_severe': '#E74C3C',
            'stress_level_critical': '#922B21'
        }
        return colors.get(self.stress_level, '#95A5A6')
    
    def get_emoji(self) -> str:
        """Get emoji for stress level"""
        emojis = {
            'stress_level_minimal': '😌',
            'stress_level_low': '🙂',
            'stress_level_moderate': '😐',
            'stress_level_high': '😟',
            'stress_level_severe': '😰',
            'stress_level_critical': '🆘'
        }
        return emojis.get(self.stress_level, '❓')
    
    def _get_domain_scores(self) -> Dict[str, float]:
        """Return domain-level certainty scores"""
        return {
            'physical_stress': self.physical_stress_cf,
            'psychological_stress': self.psychological_stress_cf,
            'lifestyle_risk': self.lifestyle_risk_cf,
            'social_isolation': self.social_isolation_cf,
            'burnout_risk': self.burnout_risk_cf,
            'anxiety_disorder': self.anxiety_disorder_risk_cf,
            'poor_sleep': self.poor_sleep_cf,
        }

    def _get_value(self, key: str, default: Any = None) -> Any:
        mapping = {
            'stress_level': self.stress_level,
            'stress_category': self.get_stress_category(),
            'stress_score': self.stress_score,
            'overall_cf': self.certainty_factor,
            'certainty_factor': self.certainty_factor,
            'fuzzy_score': self.fuzzy_score,
            'physical_stress_cf': self.physical_stress_cf,
            'psychological_stress_cf': self.psychological_stress_cf,
            'lifestyle_risk_cf': self.lifestyle_risk_cf,
            'social_isolation_cf': self.social_isolation_cf,
            'burnout_risk_cf': self.burnout_risk_cf,
            'anxiety_disorder_risk_cf': self.anxiety_disorder_risk_cf,
            'poor_sleep_cf': self.poor_sleep_cf,
            'domain_scores': self._get_domain_scores(),
            'fired_rules': self.fired_rules,
            'recommendations': self.recommendations,
            'professional_help_needed': self.professional_help_needed,
            'derived_facts': self.derived_facts,
            'stress_memberships': self.stress_memberships,
            'fired_rule_ids': self.fired_rule_ids,
            'rules_evaluated': self.rules_evaluated,
            'inference_time': self.inference_time,
        }
        return mapping.get(key, default)

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like get access for compatibility"""
        return self._get_value(key, default)

    def __getitem__(self, key: str) -> Any:
        value = self._get_value(key, None)
        if value is None and key not in {
            'stress_level', 'stress_category', 'stress_score', 'overall_cf',
            'certainty_factor', 'fuzzy_score', 'physical_stress_cf',
            'psychological_stress_cf', 'lifestyle_risk_cf',
            'social_isolation_cf', 'burnout_risk_cf',
            'anxiety_disorder_risk_cf', 'poor_sleep_cf', 'domain_scores',
            'fired_rules', 'recommendations', 'professional_help_needed',
            'derived_facts', 'stress_memberships', 'fired_rule_ids',
            'rules_evaluated', 'inference_time'
        }:
            raise KeyError(key)
        return value

    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            'stress_level': self.stress_level,
            'stress_category': self.get_stress_category(),
            'stress_score': self.stress_score,
            'overall_cf': self.certainty_factor,
            'certainty_factor': self.certainty_factor,
            'fuzzy_score': self.fuzzy_score,
            'physical_stress_cf': self.physical_stress_cf,
            'psychological_stress_cf': self.psychological_stress_cf,
            'lifestyle_risk_cf': self.lifestyle_risk_cf,
            'social_isolation_cf': self.social_isolation_cf,
            'burnout_risk_cf': self.burnout_risk_cf,
            'anxiety_disorder_risk_cf': self.anxiety_disorder_risk_cf,
            'poor_sleep_cf': self.poor_sleep_cf,
            'domain_scores': self._get_domain_scores(),
            'fired_rules': self.fired_rules,
            'recommendations': self.recommendations,
            'professional_help_needed': self.professional_help_needed,
            'derived_facts': self.derived_facts,
            'stress_memberships': self.stress_memberships,
            'fired_rule_ids': self.fired_rule_ids,
            'rules_evaluated': self.rules_evaluated,
            'inference_time': self.inference_time
        }


class InferenceEngine:
    """
    Main Inference Engine for StressSense AI
    
    Implements forward-chaining inference:
    1. Load user facts into working memory
    2. Match rules against current facts
    3. Fire matching rules, adding derived facts
    4. Repeat until no new facts can be derived
    5. Report conclusions with certainty factors
    
    Features:
    - Forward chaining with agenda
    - Certainty factor propagation
    - Fuzzy logic for soft boundaries
    - Full inference trace
    - Conflict resolution by priority
    """
    
    def __init__(self):
        self.rule_base = RuleBase()
        self.fuzzy = FuzzyMembership()
        self.cf_calc = CertaintyFactorCalculator()
        self.fuzzy_evaluator = FuzzyRuleEvaluator(threshold=0.4)
        
        # Tracking
        self.fact_base: Optional[FactBase] = None
        self.current_result: Optional[InferenceResult] = None
        
        print(f"✅ Inference Engine initialized with {self.rule_base.get_rule_count()} rules")
    
    def run(self, user_data: Dict[str, Any]) -> InferenceResult:
        """
        Main inference method
        
        Args:
            user_data: Dictionary of user inputs {fact_name: value}
        
        Returns:
            InferenceResult with complete assessment
        """
        start_time = time.time()
        
        # Initialize fresh result and fact base
        result = InferenceResult()
        self.fact_base = FactBase()
        self.current_result = result
        
        # Step 1: Load user facts into working memory
        self._load_user_facts(user_data)
        result.facts_used = len(self.fact_base)
        
        self.fact_base.add_to_trace("=== INFERENCE SESSION STARTED ===")
        self.fact_base.add_to_trace(f"Loaded {len(self.fact_base)} user facts")
        
        # Step 2: Calculate fuzzy stress score
        result.fuzzy_score = FuzzyMembership.fuzzy_stress_score(user_data)
        self.fact_base.add_to_trace(f"Fuzzy stress score: {result.fuzzy_score:.1f}/100")
        
        # Step 3: Forward chaining inference
        self._forward_chain(result)
        
        # Step 4: Determine final stress level
        self._determine_stress_level(result)
        
        # Step 5: Calculate stress score
        result.stress_score = self._calculate_stress_score(result)
        
        # Step 6: Generate recommendations
        result.recommendations = self._generate_recommendations(result, user_data)
        
        # Step 7: Check if professional help needed
        result.professional_help_needed = self._check_professional_help(result)
        
        # Step 8: Calculate fuzzy memberships for final score
        category, memberships = FuzzyMembership.get_stress_category(result.stress_score)
        result.stress_memberships = memberships
        
        # Record timing
        result.inference_time = time.time() - start_time
        
        self.fact_base.add_to_trace(f"\n=== INFERENCE COMPLETE ===")
        self.fact_base.add_to_trace(f"Stress Level: {result.stress_level}")
        self.fact_base.add_to_trace(f"Stress Score: {result.stress_score:.1f}/100")
        self.fact_base.add_to_trace(f"Inference Time: {result.inference_time:.3f}s")
        self.fact_base.add_to_trace(f"Rules Fired: {len(result.fired_rules)}")
        
        return result
    
    def _load_user_facts(self, user_data: Dict[str, Any]) -> None:
        """Load user-provided data into working memory as facts"""
        for fact_name, value in user_data.items():
            if value is not None:
                fact = Fact(
                    name=fact_name,
                    value=value,
                    fact_type=FactType.SYMPTOM,
                    certainty_factor=1.0,  # User inputs have full certainty
                    source="user_input"
                )
                self.fact_base.add_fact(fact)
    
    def _forward_chain(self, result: InferenceResult) -> None:
        """
        Perform forward chaining inference
        
        Algorithm:
        1. Check all rules against current facts
        2. Fire rules whose conditions are met
        3. Add new facts to working memory
        4. Repeat until no new rules fire (fixpoint)
        """
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        # Track which rules have been fired (avoid re-firing)
        fired_rule_ids = set()
        
        self.fact_base.add_to_trace("\n--- Forward Chaining Begins ---")
        
        while iteration < max_iterations:
            iteration += 1
            new_facts_added = False
            
            self.fact_base.add_to_trace(f"\n[Iteration {iteration}]")
            
            # Get current fact values as simple dictionary for rule evaluation
            current_facts = self._get_facts_dict()
            
            # Evaluate all rules
            rules_fired_this_iteration = 0
            
            for rule in self.rule_base.rules:
                # Skip already fired rules
                if rule.rule_id in fired_rule_ids:
                    continue
                
                result.rules_evaluated += 1
                
                # Evaluate rule conditions
                fired, cf = rule.evaluate(current_facts)
                
                if fired and abs(cf) > 0.05:  # Threshold to avoid noise
                    # Rule fires!
                    fired_rule_ids.add(rule.rule_id)
                    rules_fired_this_iteration += 1
                    
                    # Calculate conclusion CF
                    conclusion_cf = self.cf_calc.combine_with_rule_cf(1.0, cf)
                    
                    # Record fired rule
                    result.fired_rules.append({
                        'rule_id': rule.rule_id,
                        'name': rule.name,
                        'conclusion': rule.conclusion,
                        'cf': conclusion_cf,
                        'category': rule.category,
                        'explanation': rule.explanation,
                        'priority': rule.priority,
                        'iteration': iteration
                    })
                    result.fired_rule_ids.append(rule.rule_id)
                    
                    # Update domain certainty factors
                    self._update_domain_cf(result, rule.conclusion, conclusion_cf)
                    
                    # Add derived fact to working memory
                    new_facts_added = self._add_derived_fact(
                        rule.conclusion, conclusion_cf, rule.rule_id
                    )
                    
                    # Log to trace
                    self.fact_base.add_to_trace(
                        f"  ✓ FIRED {rule.rule_id}: {rule.name} "
                        f"→ {rule.conclusion} (CF={cf:.3f})"
                    )
            
            # Check for fixpoint (no new facts)
            if rules_fired_this_iteration == 0:
                self.fact_base.add_to_trace(f"  [Fixpoint reached - no new rules fire]")
                break
            
            self.fact_base.add_to_trace(
                f"  [Iteration {iteration}: {rules_fired_this_iteration} rules fired]"
            )
        
        self.fact_base.add_to_trace(
            f"\n--- Forward Chaining Complete ---"
            f"\nTotal rules fired: {len(result.fired_rules)}"
        )
    
    def _get_facts_dict(self) -> Dict[str, Any]:
        """
        Convert fact base to simple dictionary for rule evaluation
        Returns both values and derived boolean facts
        """
        facts_dict = {}
        
        # Get all fact values
        for name, fact in self.fact_base.get_all_facts().items():
            facts_dict[name] = fact.value
        
        # Add derived boolean facts based on CF thresholds
        threshold = 0.2
        
        facts_dict['physical_stress_high'] = (
            self.fact_base.get_certainty('physical_stress_high') > threshold
        )
        facts_dict['psychological_stress_high'] = (
            self.fact_base.get_certainty('psychological_stress_high') > threshold
        )
        facts_dict['lifestyle_risk_high'] = (
            self.fact_base.get_certainty('lifestyle_risk_high') > threshold
        )
        facts_dict['social_isolation_risk'] = (
            self.fact_base.get_certainty('social_isolation_risk') > threshold
        )
        facts_dict['burnout_risk'] = (
            self.fact_base.get_certainty('burnout_risk') > threshold
        )
        facts_dict['anxiety_disorder_risk'] = (
            self.fact_base.get_certainty('anxiety_disorder_risk') > threshold
        )
        facts_dict['poor_sleep'] = (
            self.fact_base.get_certainty('poor_sleep') > threshold
        )
        
        return facts_dict
    
    def _add_derived_fact(self, conclusion: str, cf: float, 
                           source_rule: str) -> bool:
        """
        Add a derived fact to the working memory
        Returns True if this is a genuinely new fact
        """
        existing_cf = self.fact_base.get_certainty(conclusion)
        
        # Combine with existing certainty
        if self.fact_base.has_fact(conclusion):
            new_cf = self.cf_calc.accumulate_evidence(existing_cf, cf)
        else:
            new_cf = cf
        
        new_cf = self.cf_calc.normalize(new_cf)
        
        # Determine if this is truly new information
        is_new = not self.fact_base.has_fact(conclusion) or abs(new_cf - existing_cf) > 0.05
        
        # Create the derived fact
        derived_fact = Fact(
            name=conclusion,
            value=new_cf > 0.2,  # True if CF exceeds threshold
            fact_type=FactType.DERIVED,
            certainty_factor=new_cf,
            source="inference",
            derived_from=[source_rule]
        )
        
        self.fact_base.add_fact(derived_fact)
        
        # Track in result
        if self.current_result:
            self.current_result.derived_facts[conclusion] = new_cf
        
        return is_new
    
    def _update_domain_cf(self, result: InferenceResult, 
                           conclusion: str, cf: float) -> None:
        """Update domain-level certainty factors in the result"""
        
        domain_mapping = {
            'physical_stress_high': 'physical',
            'psychological_stress_high': 'psychological',
            'lifestyle_risk_high': 'lifestyle',
            'social_isolation_risk': 'social',
            'burnout_risk': 'burnout',
            'anxiety_disorder_risk': 'anxiety',
            'poor_sleep': 'sleep'
        }
        
        if conclusion == 'physical_stress_high':
            result.physical_stress_cf = self.cf_calc.accumulate_evidence(
                result.physical_stress_cf, cf
            )
        elif conclusion == 'psychological_stress_high':
            result.psychological_stress_cf = self.cf_calc.accumulate_evidence(
                result.psychological_stress_cf, cf
            )
        elif conclusion == 'lifestyle_risk_high':
            result.lifestyle_risk_cf = self.cf_calc.accumulate_evidence(
                result.lifestyle_risk_cf, cf
            )
        elif conclusion == 'social_isolation_risk':
            result.social_isolation_cf = self.cf_calc.accumulate_evidence(
                result.social_isolation_cf, cf
            )
        elif conclusion == 'burnout_risk':
            result.burnout_risk_cf = self.cf_calc.accumulate_evidence(
                result.burnout_risk_cf, cf
            )
        elif conclusion == 'anxiety_disorder_risk':
            result.anxiety_disorder_risk_cf = self.cf_calc.accumulate_evidence(
                result.anxiety_disorder_risk_cf, cf
            )
        elif conclusion == 'poor_sleep':
            result.poor_sleep_cf = self.cf_calc.accumulate_evidence(
                result.poor_sleep_cf, cf
            )
    
    def _determine_stress_level(self, result: InferenceResult) -> None:
        """
        Determine final stress level from fired rules
        Uses highest-priority stress level rule that fired
        """
        stress_levels = [
            'stress_level_critical',
            'stress_level_severe', 
            'stress_level_high',
            'stress_level_moderate',
            'stress_level_low',
            'stress_level_minimal'
        ]
        
        # Find all stress level conclusions that fired
        stress_conclusions = {}
        for fired_rule in result.fired_rules:
            conclusion = fired_rule['conclusion']
            if conclusion in stress_levels:
                cf = fired_rule['cf']
                if conclusion not in stress_conclusions:
                    stress_conclusions[conclusion] = cf
                else:
                    # Accumulate evidence
                    stress_conclusions[conclusion] = self.cf_calc.accumulate_evidence(
                        stress_conclusions[conclusion], cf
                    )
        
        # If no direct stress level fired, infer from domain assessments
        if not stress_conclusions:
            result.stress_level = self._infer_stress_from_domains(result)
            return
        
        # Find highest severity level with sufficient certainty
        threshold = 0.3
        for level in stress_levels:
            if level in stress_conclusions and stress_conclusions[level] > threshold:
                result.stress_level = level
                result.certainty_factor = stress_conclusions[level]
                return
        
        # Fall back to domain inference
        result.stress_level = self._infer_stress_from_domains(result)
    
    def _infer_stress_from_domains(self, result: InferenceResult) -> str:
        """
        Infer stress level from domain certainty factors
        Used as fallback when no direct stress level rule fires
        """
        # Count how many domains are significantly stressed
        domains_stressed = sum([
            result.physical_stress_cf > 0.3,
            result.psychological_stress_cf > 0.3,
            result.lifestyle_risk_cf > 0.3,
            result.social_isolation_cf > 0.3
        ])
        
        burnout = result.burnout_risk_cf > 0.5
        anxiety = result.anxiety_disorder_risk_cf > 0.5
        
        if burnout and anxiety and domains_stressed >= 3:
            return 'stress_level_critical'
        elif burnout or (domains_stressed >= 3):
            return 'stress_level_severe'
        elif domains_stressed == 2:
            return 'stress_level_high'
        elif domains_stressed == 1:
            return 'stress_level_moderate'
        elif max(result.physical_stress_cf, result.psychological_stress_cf) > 0.2:
            return 'stress_level_low'
        else:
            return 'stress_level_minimal'
    
    def _calculate_stress_score(self, result: InferenceResult) -> float:
        """
        Calculate numerical stress score (0-100)
        Combines rule-based CF and fuzzy logic score
        """
        # Rule-based component (60% weight)
        domain_evidence = {
            'physical_stress': max(0, result.physical_stress_cf),
            'psychological_stress': max(0, result.psychological_stress_cf),
            'lifestyle_risk': max(0, result.lifestyle_risk_cf),
            'social_isolation': max(0, result.social_isolation_cf),
            'burnout_risk': max(0, result.burnout_risk_cf)
        }
        
        net_cf = self.cf_calc.calculate_net_stress_cf(domain_evidence)
        cf_score = self.cf_calc.cf_to_stress_score(net_cf)
        
        # Fuzzy component (40% weight)
        fuzzy_score = result.fuzzy_score
        
        # Level-based adjustment
        level_adjustments = {
            'stress_level_minimal': -10,
            'stress_level_low': -5,
            'stress_level_moderate': 0,
            'stress_level_high': 5,
            'stress_level_severe': 10,
            'stress_level_critical': 15
        }
        
        adjustment = level_adjustments.get(result.stress_level, 0)
        
        # Combined score
        combined = (cf_score * 0.6 + fuzzy_score * 0.4) + adjustment
        
        # Ensure stress level bounds are respected
        level_bounds = {
            'stress_level_minimal': (0, 20),
            'stress_level_low': (15, 35),
            'stress_level_moderate': (30, 55),
            'stress_level_high': (50, 72),
            'stress_level_severe': (68, 88),
            'stress_level_critical': (83, 100)
        }
        
        if result.stress_level in level_bounds:
            min_score, max_score = level_bounds[result.stress_level]
            combined = max(min_score, min(max_score, combined))
        
        return round(max(0, min(100, combined)), 1)
    
    def _generate_recommendations(self, result: InferenceResult, 
                                   user_data: Dict) -> List[str]:
        """
        Generate personalized recommendations based on assessment
        """
        recommendations = []
        
        # Sleep recommendations
        if result.poor_sleep_cf > 0.3:
            sleep_hours = user_data.get('sleep_hours', 8)
            sleep_quality = user_data.get('sleep_quality', 5)
            
            if sleep_hours < 6:
                recommendations.append(
                    "🌙 **Prioritize Sleep Duration**: You're significantly below the recommended "
                    "7-9 hours. Try going to bed 30-60 minutes earlier this week. "
                    "Consistency matters more than duration initially."
                )
            elif sleep_hours < 7:
                recommendations.append(
                    "🌙 **Improve Sleep Duration**: Aim for at least 7 hours by adjusting "
                    "your bedtime. Even 30 more minutes can make a significant difference."
                )
            
            if sleep_quality <= 4:
                recommendations.append(
                    "😴 **Enhance Sleep Quality**: Practice sleep hygiene: keep your room cool "
                    "and dark, avoid screens 1 hour before bed, and maintain consistent "
                    "sleep/wake times even on weekends."
                )
        
        # Physical stress recommendations
        if result.physical_stress_cf > 0.4:
            if user_data.get('muscle_tension', 0) >= 5:
                recommendations.append(
                    "💆 **Reduce Muscle Tension**: Try progressive muscle relaxation (PMR) - "
                    "tense and release each muscle group for 10 seconds. "
                    "Consider yoga or gentle stretching for 15 minutes daily."
                )
            
            if user_data.get('fatigue_level', 0) >= 6:
                recommendations.append(
                    "⚡ **Address Fatigue**: Schedule regular rest breaks during your day. "
                    "Consider the Pomodoro technique (25 min work, 5 min break). "
                    "Ensure you're eating regular, nutritious meals."
                )
            
            if user_data.get('heart_rate', 70) > 90:
                recommendations.append(
                    "❤️ **Monitor Heart Health**: An elevated resting heart rate suggests "
                    "ongoing stress activation. Regular aerobic exercise and stress management "
                    "can help lower resting heart rate over time."
                )
        
        # Psychological stress recommendations
        if result.psychological_stress_cf > 0.4:
            if user_data.get('anxiety_level', 0) >= 6:
                recommendations.append(
                    "🧘 **Manage Anxiety**: Practice the 4-7-8 breathing technique: "
                    "inhale for 4 counts, hold for 7, exhale for 8. "
                    "Do this 3-4 times when feeling anxious. Consider mindfulness meditation."
                )
            
            if user_data.get('overwhelm_feeling', 0) >= 6:
                recommendations.append(
                    "📋 **Combat Overwhelm**: Break large tasks into smaller steps. "
                    "Use the 'eat the frog' technique - tackle your most challenging task first. "
                    "Create a prioritized to-do list and limit it to 3 key daily tasks."
                )
            
            if user_data.get('mood_score', 5) <= 4:
                recommendations.append(
                    "😊 **Improve Mood**: Incorporate mood-boosting activities: "
                    "spend time in nature, practice gratitude journaling (3 things daily), "
                    "engage in creative activities, or connect with people you enjoy."
                )
            
            if user_data.get('concentration_ability', 5) <= 4:
                recommendations.append(
                    "🎯 **Enhance Concentration**: Try focused work sessions with "
                    "no distractions (phone away, notifications off). "
                    "Regular short breaks actually improve concentration. "
                    "Consider mindfulness practice to train attention."
                )
        
        # Lifestyle recommendations
        if result.lifestyle_risk_cf > 0.4:
            exercise = user_data.get('exercise_frequency', 3)
            if exercise < 3:
                recommendations.append(
                    "🏃 **Start Moving**: Even 20-30 minutes of moderate exercise "
                    "3 times per week significantly reduces stress hormones. "
                    "Start with walks - the outdoors adds additional stress-relief benefits."
                )
            
            if user_data.get('work_hours', 8) >= 10:
                recommendations.append(
                    "⏰ **Set Work Boundaries**: Overworking is a major stress factor. "
                    "Define clear work start/end times. Use time-blocking to be more efficient. "
                    "Take at least a 30-minute lunch break away from work."
                )
            
            if user_data.get('caffeine_intake', 'moderate') in ['high', 'very_high']:
                recommendations.append(
                    "☕ **Reduce Caffeine**: High caffeine amplifies the stress response. "
                    "Gradually reduce intake (avoid cold turkey to prevent headaches). "
                    "Replace afternoon coffee with herbal tea or water."
                )
            
            if user_data.get('relaxation_activities', 'sometimes') in ['never', 'rarely']:
                recommendations.append(
                    "🎯 **Schedule Relaxation**: Relaxation isn't optional - it's essential. "
                    "Schedule 15-30 minutes of relaxation daily. Options: reading, hobbies, "
                    "music, nature walks, meditation, or any activity you enjoy."
                )
        
        # Social recommendations
        if result.social_isolation_cf > 0.4:
            recommendations.append(
                "👥 **Build Social Connection**: Social support is one of the strongest "
                "stress buffers. Reach out to a friend or family member today. "
                "Consider joining a group with shared interests, or volunteering."
            )
            
            if user_data.get('relationship_quality', 'good') in ['poor', 'very_poor']:
                recommendations.append(
                    "💬 **Improve Relationships**: Consider having an open conversation "
                    "with important people in your life. If needed, relationship counseling "
                    "can help. Focus on one relationship at a time."
                )
        
        # Burnout recommendations
        if result.burnout_risk_cf > 0.5:
            recommendations.append(
                "🚨 **Address Burnout Risk**: You show signs of burnout. "
                "This requires immediate action: 1) Take time off if possible, "
                "2) Identify and reduce your biggest stressors, "
                "3) Seek support from your manager/HR about workload, "
                "4) Consider speaking with a professional counselor."
            )
        
        # Financial stress recommendations
        if user_data.get('financial_stress', 0) >= 7:
            recommendations.append(
                "💰 **Manage Financial Stress**: Create a simple budget to understand "
                "your situation clearly. Contact a free financial counselor if needed. "
                "Focus on what you can control - small financial wins reduce anxiety."
            )
        
        # Positive reinforcement
        if result.physical_stress_cf < 0.3 and result.psychological_stress_cf < 0.3:
            recommendations.append(
                "✅ **Maintain Your Healthy Habits**: Your current lifestyle shows good "
                "stress management. Continue with your exercise routine, sleep habits, "
                "and relaxation practices. Regular check-ins help maintain this balance."
            )
        
        # General recommendations always included
        recommendations.append(
            "📊 **Track Your Progress**: Consider keeping a brief daily stress journal. "
            "Note your stress triggers, what helps, and your mood. "
            "This self-awareness is powerful for long-term stress management."
        )
        
        return recommendations[:8]  # Return top 8 recommendations
    
    def _check_professional_help(self, result: InferenceResult) -> bool:
        """Determine if professional help should be recommended"""
        return (
            result.stress_level in ['stress_level_severe', 'stress_level_critical'] or
            result.burnout_risk_cf > 0.7 or
            result.anxiety_disorder_risk_cf > 0.7 or
            result.stress_score >= 75
        )
    
    def get_inference_trace(self) -> List[str]:
        """Get the complete inference trace"""
        if self.fact_base:
            return self.fact_base.get_trace()
        return []
    
    def get_rule_base_summary(self) -> Dict:
        """Get summary of the rule base"""
        return self.rule_base.get_summary()
    
    def get_fired_rules_by_category(self, result: InferenceResult) -> Dict[str, List]:
        """Organize fired rules by category"""
        by_category = {}
        for rule_info in result.fired_rules:
            category = rule_info.get('category', 'unknown')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(rule_info)
        return by_category