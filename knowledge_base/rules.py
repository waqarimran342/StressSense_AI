"""
Rule Base - All 65+ Inference Rules for StressSense AI
Rules use certainty factors (CF) following MYCIN methodology
Rule format: IF conditions THEN conclusion (with CF)
"""

from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Rule:
    """
    Represents a single inference rule
    
    Attributes:
        rule_id: Unique identifier (e.g., "R001")
        name: Human-readable name
        description: What this rule detects
        category: Rule category for organization
        conditions: Function that evaluates conditions
        conclusion: What to conclude if conditions met
        certainty_factor: Confidence in this rule (0.0-1.0)
        priority: Rule firing priority (higher = fires first)
        explanation: Human-readable explanation for WHY/HOW
    """
    rule_id: str
    name: str
    description: str
    category: str
    conditions: Callable
    conclusion: str
    certainty_factor: float
    priority: int = 1
    explanation: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def evaluate(self, facts: Dict) -> Tuple[bool, float]:
        """
        Evaluate rule conditions against current facts
        Returns (fired, certainty_factor)
        """
        try:
            result = self.conditions(facts)
            if isinstance(result, tuple):
                return result
            return (bool(result), self.certainty_factor if result else 0.0)
        except (KeyError, TypeError, ZeroDivisionError):
            return (False, 0.0)
    
    def __repr__(self) -> str:
        return f"Rule({self.rule_id}: {self.name}, CF={self.certainty_factor})"


class RuleBase:
    """
    Complete Rule Base for StressSense AI
    Contains 65+ rules organized by category
    """
    
    def __init__(self):
        self.rules: List[Rule] = []
        self._initialize_all_rules()
    
    def _initialize_all_rules(self):
        """Initialize all rules in the knowledge base"""
        self._add_sleep_rules()
        self._add_physiological_rules()
        self._add_psychological_rules()
        self._add_lifestyle_rules()
        self._add_environmental_rules()
        self._add_social_rules()
        self._add_burnout_rules()
        self._add_anxiety_rules()
        self._add_composite_stress_rules()
        self._add_stress_level_rules()
        self._add_protective_factor_rules()
        
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    # ============================================================
    # SLEEP RULES (R001-R010)
    # ============================================================
    
    def _add_sleep_rules(self):
        """Rules related to sleep patterns"""
        
        # R001: Severe sleep deprivation
        self.rules.append(Rule(
            rule_id="R001",
            name="Severe Sleep Deprivation",
            description="Less than 5 hours sleep indicates severe deprivation",
            category="sleep",
            conditions=lambda f: (
                f.get('sleep_hours', 8) < 5,
                0.90
            ),
            conclusion="poor_sleep",
            certainty_factor=0.90,
            priority=10,
            explanation="Sleeping less than 5 hours per night is severely below the recommended 7-9 hours. This strongly indicates sleep deprivation which is a major stress factor."
        ))
        
        # R002: Moderate sleep deprivation
        self.rules.append(Rule(
            rule_id="R002",
            name="Moderate Sleep Deprivation",
            description="5-6 hours sleep indicates moderate deprivation",
            category="sleep",
            conditions=lambda f: (
                5 <= f.get('sleep_hours', 8) < 6.5,
                0.70
            ),
            conclusion="poor_sleep",
            certainty_factor=0.70,
            priority=9,
            explanation="Sleeping 5-6.5 hours is below the recommended amount. Chronic sleep restriction accumulates a 'sleep debt' that significantly impacts stress responses."
        ))
        
        # R003: Borderline sleep
        self.rules.append(Rule(
            rule_id="R003",
            name="Borderline Insufficient Sleep",
            description="6.5-7 hours may be insufficient for some people",
            category="sleep",
            conditions=lambda f: (
                6.5 <= f.get('sleep_hours', 8) < 7,
                0.40
            ),
            conclusion="poor_sleep",
            certainty_factor=0.40,
            priority=8,
            explanation="While 6.5-7 hours is close to the minimum recommendation, many people need 7-9 hours for optimal functioning."
        ))
        
        # R004: Poor sleep quality
        self.rules.append(Rule(
            rule_id="R004",
            name="Poor Sleep Quality",
            description="Low sleep quality score indicates restorative sleep issues",
            category="sleep",
            conditions=lambda f: (
                f.get('sleep_quality', 5) <= 3,
                0.85
            ),
            conclusion="poor_sleep",
            certainty_factor=0.85,
            priority=10,
            explanation="Sleep quality of 3/10 or below indicates highly non-restorative sleep. Even sufficient hours of poor-quality sleep can lead to stress and fatigue."
        ))
        
        # R005: Moderate sleep quality issues
        self.rules.append(Rule(
            rule_id="R005",
            name="Moderate Sleep Quality Issues",
            description="Sleep quality 4-5 indicates moderate issues",
            category="sleep",
            conditions=lambda f: (
                3 < f.get('sleep_quality', 5) <= 5,
                0.60
            ),
            conclusion="poor_sleep",
            certainty_factor=0.60,
            priority=8,
            explanation="Sleep quality rated 4-5 out of 10 suggests moderately disrupted sleep patterns, which can contribute to daytime stress and fatigue."
        ))
        
        # R006: Combined sleep quantity and quality problem
        self.rules.append(Rule(
            rule_id="R006",
            name="Combined Sleep Deficit",
            description="Both low hours AND low quality creates compound sleep problem",
            category="sleep",
            conditions=lambda f: (
                f.get('sleep_hours', 8) < 6.5 and 
                f.get('sleep_quality', 5) <= 4,
                0.95
            ),
            conclusion="poor_sleep",
            certainty_factor=0.95,
            priority=11,
            explanation="Both insufficient sleep duration AND poor quality create a severe compound sleep deficit. This is one of the strongest predictors of high stress."
        ))
        
        # R007: Sleep affecting physical stress
        self.rules.append(Rule(
            rule_id="R007",
            name="Sleep-Physical Stress Connection",
            description="Poor sleep combined with high fatigue indicates physical stress",
            category="sleep",
            conditions=lambda f: (
                f.get('sleep_hours', 8) < 6.5 and 
                f.get('fatigue_level', 0) >= 6,
                0.80
            ),
            conclusion="physical_stress_high",
            certainty_factor=0.80,
            priority=9,
            explanation="The combination of poor sleep and high fatigue levels strongly indicates elevated physical stress. The body's recovery mechanisms are overwhelmed."
        ))
        
        # R008: Excessive sleep (possible depression indicator)
        self.rules.append(Rule(
            rule_id="R008",
            name="Excessive Sleep Pattern",
            description="Sleeping more than 10 hours may indicate depression/stress",
            category="sleep",
            conditions=lambda f: (
                f.get('sleep_hours', 8) > 10,
                0.50
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.50,
            priority=7,
            explanation="Consistently sleeping more than 10 hours can indicate depression or avoidance behavior associated with high psychological stress."
        ))
        
        # R009: Sleep quality and psychological stress
        self.rules.append(Rule(
            rule_id="R009",
            name="Sleep Quality Psychological Impact",
            description="Very poor sleep quality linked to psychological stress",
            category="sleep",
            conditions=lambda f: (
                f.get('sleep_quality', 5) <= 3 and 
                f.get('anxiety_level', 0) >= 5,
                0.85
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.85,
            priority=10,
            explanation="Poor sleep quality combined with elevated anxiety creates a reinforcing cycle - anxiety disrupts sleep, and poor sleep increases anxiety."
        ))
        
        # R010: Optimal sleep (protective)
        self.rules.append(Rule(
            rule_id="R010",
            name="Adequate Sleep Protection",
            description="Good sleep quality and duration provides stress protection",
            category="sleep",
            conditions=lambda f: (
                f.get('sleep_hours', 8) >= 7 and 
                f.get('sleep_hours', 8) <= 9 and
                f.get('sleep_quality', 5) >= 7,
                -0.40  # Negative CF = reduces stress
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=-0.40,
            priority=8,
            explanation="Good sleep (7-9 hours of quality sleep) is one of the most important protective factors against stress. This significantly reduces stress risk."
        ))
    
    # ============================================================
    # PHYSIOLOGICAL RULES (R011-R021)
    # ============================================================
    
    def _add_physiological_rules(self):
        """Rules related to physical symptoms"""
        
        # R011: High resting heart rate
        self.rules.append(Rule(
            rule_id="R011",
            name="Elevated Resting Heart Rate",
            description="Heart rate above 100 bpm at rest suggests stress",
            category="physiological",
            conditions=lambda f: (
                f.get('heart_rate', 70) > 100,
                0.80
            ),
            conclusion="physical_stress_high",
            certainty_factor=0.80,
            priority=9,
            explanation="A resting heart rate above 100 bpm (tachycardia) is a physiological indicator of stress. The sympathetic nervous system ('fight or flight') is chronically activated."
        ))
        
        # R012: Moderately elevated heart rate
        self.rules.append(Rule(
            rule_id="R012",
            name="Moderately Elevated Heart Rate",
            description="Heart rate 80-100 bpm may indicate stress activation",
            category="physiological",
            conditions=lambda f: (
                80 < f.get('heart_rate', 70) <= 100,
                0.50
            ),
            conclusion="physical_stress_high",
            certainty_factor=0.50,
            priority=7,
            explanation="Heart rate between 80-100 bpm at rest is on the higher end of normal and may indicate mild stress response or lifestyle factors."
        ))
        
        # R013: Frequent headaches
        self.rules.append(Rule(
            rule_id="R013",
            name="Frequent Headaches",
            description="Frequent headaches are a common stress symptom",
            category="physiological",
            conditions=lambda f: (
                f.get('headache_frequency', 'never') in ['often', 'always'],
                0.75
            ),
            conclusion="physical_stress_high",
            certainty_factor=0.75,
            priority=8,
            explanation="Tension headaches occurring often or always are a classic physical manifestation of stress. Stress causes muscle tension around the head and neck."
        ))
        
        # R014: Occasional headaches
        self.rules.append(Rule(
            rule_id="R014",
            name="Occasional Headaches",
            description="Sometimes having headaches mildly indicates stress",
            category="physiological",
            conditions=lambda f: (
                f.get('headache_frequency', 'never') == 'sometimes',
                0.45
            ),
            conclusion="physical_stress_high",
            certainty_factor=0.45,
            priority=6,
            explanation="Occasional headaches can be associated with stress, particularly tension-type headaches that occur during stressful periods."
        ))
        
        # R015: High muscle tension
        self.rules.append(Rule(
            rule_id="R015",
            name="Significant Muscle Tension",
            description="High muscle tension is a primary stress indicator",
            category="physiological",
            conditions=lambda f: (
                f.get('muscle_tension', 0) >= 7,
                0.80
            ),
            conclusion="physical_stress_high",
            certainty_factor=0.80,
            priority=9,
            explanation="Muscle tension of 7/10 or higher indicates the body is in a prolonged state of tension. This is directly linked to stress hormones like cortisol."
        ))
        
        # R016: Moderate muscle tension
        self.rules.append(Rule(
            rule_id="R016",
            name="Moderate Muscle Tension",
            description="Moderate muscle tension indicates some physical stress",
            category="physiological",
            conditions=lambda f: (
                4 <= f.get('muscle_tension', 0) < 7,
                0.50
            ),
            conclusion="physical_stress_high",
            certainty_factor=0.50,
            priority=7,
            explanation="Moderate muscle tension (4-7/10) is a common stress response, particularly in neck, shoulders, and back muscles."
        ))
        
        # R017: High fatigue
        self.rules.append(Rule(
            rule_id="R017",
            name="Severe Physical Fatigue",
            description="Very high fatigue levels indicate physical stress burden",
            category="physiological",
            conditions=lambda f: (
                f.get('fatigue_level', 0) >= 8,
                0.80
            ),
            conclusion="physical_stress_high",
            certainty_factor=0.80,
            priority=9,
            explanation="Fatigue levels of 8/10 or higher indicate the body's resources are significantly depleted. This severe fatigue is both a cause and consequence of chronic stress."
        ))
        
        # R018: Moderate fatigue
        self.rules.append(Rule(
            rule_id="R018",
            name="Moderate Fatigue",
            description="Moderate fatigue contributes to stress",
            category="physiological",
            conditions=lambda f: (
                5 <= f.get('fatigue_level', 0) < 8,
                0.55
            ),
            conclusion="physical_stress_high",
            certainty_factor=0.55,
            priority=7,
            explanation="Moderate fatigue (5-8/10) indicates the body is working harder than comfortable, often associated with stress and recovery deficits."
        ))
        
        # R019: Appetite disruption
        self.rules.append(Rule(
            rule_id="R019",
            name="Significant Appetite Disruption",
            description="Severely decreased appetite indicates stress impact on body",
            category="physiological",
            conditions=lambda f: (
                f.get('appetite_change', 'normal') == 'severely_decreased',
                0.75
            ),
            conclusion="physical_stress_high",
            certainty_factor=0.75,
            priority=8,
            explanation="Severely decreased appetite is a significant physical stress response. Stress hormones like cortisol and adrenaline suppress hunger signals."
        ))
        
        # R020: Digestive issues
        self.rules.append(Rule(
            rule_id="R020",
            name="Frequent Digestive Issues",
            description="Chronic digestive problems linked to stress",
            category="physiological",
            conditions=lambda f: (
                f.get('digestive_issues', 'never') in ['often', 'always'],
                0.70
            ),
            conclusion="physical_stress_high",
            certainty_factor=0.70,
            priority=8,
            explanation="Frequent digestive issues are directly linked to stress through the gut-brain axis. Stress disrupts normal digestive function through the autonomic nervous system."
        ))
        
        # R021: Multiple physical symptoms
        self.rules.append(Rule(
            rule_id="R021",
            name="Multiple Physical Stress Symptoms",
            description="Combination of physical symptoms strongly indicates high stress",
            category="physiological",
            conditions=lambda f: (
                (f.get('headache_frequency', 'never') in ['often', 'always']) and
                (f.get('muscle_tension', 0) >= 6) and
                (f.get('fatigue_level', 0) >= 6),
                0.90
            ),
            conclusion="physical_stress_high",
            certainty_factor=0.90,
            priority=11,
            explanation="The presence of multiple physical stress symptoms simultaneously (headaches + muscle tension + fatigue) strongly confirms significant physical stress response."
        ))
    
    # ============================================================
    # PSYCHOLOGICAL RULES (R022-R033)
    # ============================================================
    
    def _add_psychological_rules(self):
        """Rules related to psychological symptoms"""
        
        # R022: Severe anxiety
        self.rules.append(Rule(
            rule_id="R022",
            name="Severe Anxiety",
            description="Anxiety level 8+ indicates severe psychological stress",
            category="psychological",
            conditions=lambda f: (
                f.get('anxiety_level', 0) >= 8,
                0.90
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.90,
            priority=10,
            explanation="An anxiety level of 8/10 or higher is a strong indicator of severe psychological stress. This level of anxiety significantly impairs daily functioning."
        ))
        
        # R023: Moderate to high anxiety
        self.rules.append(Rule(
            rule_id="R023",
            name="Significant Anxiety",
            description="Anxiety level 6-7 indicates significant psychological stress",
            category="psychological",
            conditions=lambda f: (
                6 <= f.get('anxiety_level', 0) < 8,
                0.70
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.70,
            priority=9,
            explanation="Anxiety levels of 6-7/10 represent significant anxiety that frequently interferes with daily activities and relationships."
        ))
        
        # R024: Moderate anxiety
        self.rules.append(Rule(
            rule_id="R024",
            name="Moderate Anxiety",
            description="Anxiety level 4-5 indicates moderate psychological concern",
            category="psychological",
            conditions=lambda f: (
                4 <= f.get('anxiety_level', 0) < 6,
                0.45
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.45,
            priority=7,
            explanation="Moderate anxiety (4-5/10) affects quality of life and may interfere with sleep, relationships, and work performance."
        ))
        
        # R025: Poor mood
        self.rules.append(Rule(
            rule_id="R025",
            name="Persistently Poor Mood",
            description="Low mood score indicates psychological distress",
            category="psychological",
            conditions=lambda f: (
                f.get('mood_score', 5) <= 3,
                0.80
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.80,
            priority=9,
            explanation="A mood score of 3/10 or below indicates significant psychological distress. Persistent low mood is both a symptom and amplifier of stress."
        ))
        
        # R026: Moderate mood issues
        self.rules.append(Rule(
            rule_id="R026",
            name="Below Average Mood",
            description="Mood score 4-5 suggests elevated psychological stress",
            category="psychological",
            conditions=lambda f: (
                3 < f.get('mood_score', 5) <= 5,
                0.55
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.55,
            priority=7,
            explanation="Below-average mood (4-5/10) consistently reported suggests chronic stress is affecting emotional wellbeing."
        ))
        
        # R027: Concentration difficulties
        self.rules.append(Rule(
            rule_id="R027",
            name="Significant Concentration Problems",
            description="Low concentration ability indicates cognitive stress impact",
            category="psychological",
            conditions=lambda f: (
                f.get('concentration_ability', 5) <= 3,
                0.75
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.75,
            priority=8,
            explanation="Difficulty concentrating (3/10 or below) is a key cognitive symptom of stress. Stress hormones impair prefrontal cortex function, affecting attention and memory."
        ))
        
        # R028: Moderate concentration issues
        self.rules.append(Rule(
            rule_id="R028",
            name="Moderate Concentration Difficulties",
            description="Moderate concentration issues suggest stress impact on cognition",
            category="psychological",
            conditions=lambda f: (
                3 < f.get('concentration_ability', 5) <= 5,
                0.50
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.50,
            priority=7,
            explanation="Moderate concentration difficulties (4-5/10) indicate stress is beginning to affect cognitive performance and productivity."
        ))
        
        # R029: High irritability
        self.rules.append(Rule(
            rule_id="R029",
            name="High Irritability",
            description="High irritability is a key stress indicator",
            category="psychological",
            conditions=lambda f: (
                f.get('irritability_level', 0) >= 7,
                0.80
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.80,
            priority=9,
            explanation="High irritability (7/10 or above) is a hallmark stress symptom. Stress reduces emotional regulation capacity, lowering the threshold for frustration responses."
        ))
        
        # R030: High overwhelm feeling
        self.rules.append(Rule(
            rule_id="R030",
            name="Severe Overwhelm Feeling",
            description="Feeling severely overwhelmed indicates high stress",
            category="psychological",
            conditions=lambda f: (
                f.get('overwhelm_feeling', 0) >= 7,
                0.85
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.85,
            priority=10,
            explanation="Feeling overwhelmed at 7/10 or above indicates the perceived demands exceed available coping resources - the core definition of psychological stress."
        ))
        
        # R031: Low motivation
        self.rules.append(Rule(
            rule_id="R031",
            name="Significantly Low Motivation",
            description="Very low motivation linked to stress and burnout",
            category="psychological",
            conditions=lambda f: (
                f.get('motivation_level', 5) <= 3,
                0.70
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.70,
            priority=8,
            explanation="Low motivation (3/10 or below) is associated with chronic stress and early burnout. Sustained high stress depletes the brain's reward and motivation systems."
        ))
        
        # R032: Frequent worry
        self.rules.append(Rule(
            rule_id="R032",
            name="Excessive Worry Pattern",
            description="Frequent/constant worry indicates anxiety-based stress",
            category="psychological",
            conditions=lambda f: (
                f.get('worry_frequency', 'never') in ['often', 'always'],
                0.75
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.75,
            priority=8,
            explanation="Frequent or constant worrying is characteristic of anxiety-related stress. Rumination and worry perpetuate the stress response even without external stressors."
        ))
        
        # R033: Multiple psychological symptoms
        self.rules.append(Rule(
            rule_id="R033",
            name="Multiple Psychological Stress Indicators",
            description="Multiple psychological symptoms confirm high psychological stress",
            category="psychological",
            conditions=lambda f: (
                f.get('anxiety_level', 0) >= 6 and
                f.get('mood_score', 5) <= 4 and
                f.get('overwhelm_feeling', 0) >= 6,
                0.92
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.92,
            priority=11,
            explanation="The combination of high anxiety, low mood, and feeling overwhelmed is a powerful indicator of significant psychological stress requiring attention."
        ))
    
    # ============================================================
    # LIFESTYLE RULES (R034-R044)
    # ============================================================
    
    def _add_lifestyle_rules(self):
        """Rules related to lifestyle factors"""
        
        # R034: Sedentary lifestyle
        self.rules.append(Rule(
            rule_id="R034",
            name="Sedentary Lifestyle",
            description="No exercise significantly increases stress vulnerability",
            category="lifestyle",
            conditions=lambda f: (
                f.get('exercise_frequency', 3) == 0,
                0.65
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.65,
            priority=8,
            explanation="No regular exercise removes one of the most effective natural stress-relief mechanisms. Exercise releases endorphins and reduces cortisol levels."
        ))
        
        # R035: Very low exercise
        self.rules.append(Rule(
            rule_id="R035",
            name="Insufficient Exercise",
            description="Less than 2 days/week exercise increases stress risk",
            category="lifestyle",
            conditions=lambda f: (
                0 < f.get('exercise_frequency', 3) < 2,
                0.50
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.50,
            priority=7,
            explanation="Exercising less than twice a week is insufficient to gain significant stress-protective benefits from physical activity."
        ))
        
        # R036: Overworking
        self.rules.append(Rule(
            rule_id="R036",
            name="Excessive Work Hours",
            description="Working 10+ hours daily is a major stress risk factor",
            category="lifestyle",
            conditions=lambda f: (
                f.get('work_hours', 8) >= 10,
                0.80
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.80,
            priority=9,
            explanation="Working 10 or more hours daily significantly increases stress risk. Extended work hours leave insufficient time for recovery, social connection, and self-care."
        ))
        
        # R037: Long work hours
        self.rules.append(Rule(
            rule_id="R037",
            name="Extended Work Hours",
            description="Working 8-10 hours daily increases stress risk",
            category="lifestyle",
            conditions=lambda f: (
                8 <= f.get('work_hours', 8) < 10,
                0.50
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.50,
            priority=7,
            explanation="Consistently working 8-10 hours daily, especially without breaks, can contribute to stress accumulation and poor work-life balance."
        ))
        
        # R038: High caffeine intake
        self.rules.append(Rule(
            rule_id="R038",
            name="High Caffeine Consumption",
            description="High/very high caffeine amplifies stress response",
            category="lifestyle",
            conditions=lambda f: (
                f.get('caffeine_intake', 'moderate') in ['high', 'very_high'],
                0.60
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.60,
            priority=7,
            explanation="High caffeine intake stimulates the sympathetic nervous system, mimicking and amplifying the stress response. It also disrupts sleep quality."
        ))
        
        # R039: Alcohol consumption
        self.rules.append(Rule(
            rule_id="R039",
            name="Frequent Alcohol Use",
            description="Frequent/heavy alcohol use indicates stress coping issues",
            category="lifestyle",
            conditions=lambda f: (
                f.get('alcohol_consumption', 'occasional') in ['frequent', 'heavy'],
                0.70
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.70,
            priority=8,
            explanation="Frequent or heavy alcohol consumption is often used as a stress coping mechanism but actually increases stress long-term by disrupting sleep and neurochemistry."
        ))
        
        # R040: Regular smoking
        self.rules.append(Rule(
            rule_id="R040",
            name="Regular Smoking",
            description="Regular smoking both reflects and amplifies stress",
            category="lifestyle",
            conditions=lambda f: (
                f.get('smoking_status', 'never') == 'regular',
                0.60
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.60,
            priority=7,
            explanation="Regular smoking is associated with stress - while smokers report temporary relief, nicotine withdrawal between cigarettes actually increases baseline stress levels."
        ))
        
        # R041: No relaxation activities
        self.rules.append(Rule(
            rule_id="R041",
            name="Absence of Relaxation Activities",
            description="Never engaging in relaxation increases stress accumulation",
            category="lifestyle",
            conditions=lambda f: (
                f.get('relaxation_activities', 'sometimes') == 'never',
                0.65
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.65,
            priority=8,
            explanation="Never engaging in relaxation activities means stress has no outlet or recovery mechanism. Relaxation activities are essential for resetting the stress response."
        ))
        
        # R042: Excessive screen time
        self.rules.append(Rule(
            rule_id="R042",
            name="Excessive Screen Time",
            description="More than 6 hours recreational screen time increases stress",
            category="lifestyle",
            conditions=lambda f: (
                f.get('screen_time_hours', 3) > 6,
                0.50
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.50,
            priority=6,
            explanation="Excessive screen time (6+ hours recreationally) is associated with increased stress, disrupted sleep due to blue light, and reduced time for restorative activities."
        ))
        
        # R043: Poor work-life balance
        self.rules.append(Rule(
            rule_id="R043",
            name="Poor Work-Life Balance",
            description="Very poor work-life balance is a major lifestyle stressor",
            category="lifestyle",
            conditions=lambda f: (
                f.get('work_life_balance', 5) <= 3,
                0.75
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.75,
            priority=9,
            explanation="Poor work-life balance (3/10 or below) means insufficient time for recovery, personal relationships, and activities that provide meaning and joy outside of work."
        ))
        
        # R044: Protective lifestyle factors
        self.rules.append(Rule(
            rule_id="R044",
            name="Healthy Lifestyle Practices",
            description="Regular exercise and relaxation reduce stress",
            category="lifestyle",
            conditions=lambda f: (
                f.get('exercise_frequency', 0) >= 4 and
                f.get('relaxation_activities', 'never') in ['often', 'daily'],
                -0.45
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=-0.45,
            priority=8,
            explanation="Regular exercise (4+ days/week) combined with frequent relaxation activities provides significant protection against stress. These are among the most effective stress management strategies."
        ))
    
    # ============================================================
    # ENVIRONMENTAL RULES (R045-R051)
    # ============================================================
    
    def _add_environmental_rules(self):
        """Rules related to environmental factors"""
        
        # R045: Negative work environment
        self.rules.append(Rule(
            rule_id="R045",
            name="Negative Work Environment",
            description="Very negative work environment is a major stressor",
            category="environmental",
            conditions=lambda f: (
                f.get('work_environment', 'neutral') == 'very_negative',
                0.85
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.85,
            priority=9,
            explanation="A very negative work environment is one of the strongest occupational stressors. This includes toxic cultures, harassment, excessive demands, and lack of control."
        ))
        
        # R046: Poor work environment
        self.rules.append(Rule(
            rule_id="R046",
            name="Poor Work Environment",
            description="Negative work environment contributes to stress",
            category="environmental",
            conditions=lambda f: (
                f.get('work_environment', 'neutral') == 'negative',
                0.65
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.65,
            priority=8,
            explanation="A negative work environment significantly contributes to chronic stress. Poor management, unclear expectations, and conflict all elevate baseline stress."
        ))
        
        # R047: Multiple life changes
        self.rules.append(Rule(
            rule_id="R047",
            name="Multiple Major Life Changes",
            description="3+ major life changes in 6 months creates high stress",
            category="environmental",
            conditions=lambda f: (
                f.get('major_life_changes', 0) >= 3,
                0.75
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.75,
            priority=8,
            explanation="Three or more major life changes (moving, job change, relationships, loss) in 6 months creates a high cumulative stress load, as each change requires significant adaptation."
        ))
        
        # R048: High financial stress
        self.rules.append(Rule(
            rule_id="R048",
            name="Significant Financial Stress",
            description="High financial stress is a major environmental stressor",
            category="environmental",
            conditions=lambda f: (
                f.get('financial_stress', 0) >= 7,
                0.80
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.80,
            priority=9,
            explanation="High financial stress (7/10+) is one of the most persistent and impactful stressors. Financial worry activates the stress response and disrupts sleep and relationships."
        ))
        
        # R049: Moderate financial stress
        self.rules.append(Rule(
            rule_id="R049",
            name="Moderate Financial Stress",
            description="Moderate financial stress contributes to overall stress",
            category="environmental",
            conditions=lambda f: (
                4 <= f.get('financial_stress', 0) < 7,
                0.50
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.50,
            priority=7,
            explanation="Moderate financial stress (4-7/10) creates persistent background anxiety that contributes to overall stress levels."
        ))
        
        # R050: Unsatisfactory living situation
        self.rules.append(Rule(
            rule_id="R050",
            name="Poor Living Situation",
            description="Very unsatisfied with living situation adds significant stress",
            category="environmental",
            conditions=lambda f: (
                f.get('living_situation', 'neutral') in ['unsatisfied', 'very_unsatisfied'],
                0.60
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=0.60,
            priority=7,
            explanation="Dissatisfaction with one's living situation (housing insecurity, noise, conflict) is a chronic environmental stressor that affects wellbeing continuously."
        ))
        
        # R051: Positive environment (protective)
        self.rules.append(Rule(
            rule_id="R051",
            name="Positive Environmental Factors",
            description="Good work environment and stable life reduces stress",
            category="environmental",
            conditions=lambda f: (
                f.get('work_environment', 'neutral') in ['positive', 'very_positive'] and
                f.get('financial_stress', 5) <= 3,
                -0.35
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=-0.35,
            priority=7,
            explanation="A positive work environment combined with manageable financial stress provides significant environmental protection against chronic stress."
        ))
    
    # ============================================================
    # SOCIAL RULES (R052-R057)
    # ============================================================
    
    def _add_social_rules(self):
        """Rules related to social factors"""
        
        # R052: Low social support
        self.rules.append(Rule(
            rule_id="R052",
            name="Inadequate Social Support",
            description="Low social support is a major stress risk factor",
            category="social",
            conditions=lambda f: (
                f.get('social_support', 5) <= 3,
                0.75
            ),
            conclusion="social_isolation_risk",
            certainty_factor=0.75,
            priority=9,
            explanation="Low social support (3/10 or below) removes one of the most powerful buffers against stress. Social connection is fundamental to stress resilience."
        ))
        
        # R053: Social isolation feeling
        self.rules.append(Rule(
            rule_id="R053",
            name="High Isolation Feeling",
            description="Strong feeling of isolation significantly increases stress",
            category="social",
            conditions=lambda f: (
                f.get('isolation_feeling', 0) >= 7,
                0.80
            ),
            conclusion="social_isolation_risk",
            certainty_factor=0.80,
            priority=9,
            explanation="Feeling highly isolated (7/10+) is both a consequence and cause of stress. Loneliness activates the same neural pathways as physical pain and chronic stress."
        ))
        
        # R054: Poor relationships
        self.rules.append(Rule(
            rule_id="R054",
            name="Poor Relationship Quality",
            description="Poor close relationships remove key stress buffer",
            category="social",
            conditions=lambda f: (
                f.get('relationship_quality', 'good') in ['poor', 'very_poor'],
                0.70
            ),
            conclusion="social_isolation_risk",
            certainty_factor=0.70,
            priority=8,
            explanation="Poor quality close relationships (poor/very poor) remove important social support while potentially adding conflict-related stress."
        ))
        
        # R055: Social isolation linked to psychological stress
        self.rules.append(Rule(
            rule_id="R055",
            name="Social Isolation Psychological Impact",
            description="Social isolation combined with mood issues increases psychological stress",
            category="social",
            conditions=lambda f: (
                f.get('isolation_feeling', 0) >= 6 and
                f.get('mood_score', 5) <= 4,
                0.85
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.85,
            priority=10,
            explanation="The combination of social isolation and low mood creates a compounding psychological stress effect. Social connection is essential for emotional regulation."
        ))
        
        # R056: Strong social support (protective)
        self.rules.append(Rule(
            rule_id="R056",
            name="Strong Social Support Network",
            description="High social support provides significant stress protection",
            category="social",
            conditions=lambda f: (
                f.get('social_support', 5) >= 8 and
                f.get('relationship_quality', 'good') in ['excellent', 'good'],
                -0.50
            ),
            conclusion="social_isolation_risk",
            certainty_factor=-0.50,
            priority=9,
            explanation="A strong social support network with quality relationships is one of the most powerful stress protective factors, buffering against almost all stress types."
        ))
        
        # R057: Moderate social isolation
        self.rules.append(Rule(
            rule_id="R057",
            name="Moderate Social Isolation",
            description="Moderate isolation feeling increases stress risk",
            category="social",
            conditions=lambda f: (
                4 <= f.get('isolation_feeling', 0) < 7,
                0.50
            ),
            conclusion="social_isolation_risk",
            certainty_factor=0.50,
            priority=7,
            explanation="Moderate feelings of social isolation (4-7/10) indicate insufficient social connection that can gradually increase stress vulnerability."
        ))
    
    # ============================================================
    # BURNOUT RULES (R058-R061)
    # ============================================================
    
    def _add_burnout_rules(self):
        """Rules specifically for detecting burnout risk"""
        
        # R058: Classic burnout pattern
        self.rules.append(Rule(
            rule_id="R058",
            name="Classic Burnout Indicators",
            description="Exhaustion + cynicism + low efficacy = burnout pattern",
            category="burnout",
            conditions=lambda f: (
                f.get('fatigue_level', 0) >= 7 and
                f.get('motivation_level', 5) <= 3 and
                f.get('work_hours', 8) >= 9,
                0.88
            ),
            conclusion="burnout_risk",
            certainty_factor=0.88,
            priority=11,
            explanation="The combination of high fatigue (7+/10), very low motivation (3-/10), and long work hours (9+/day) matches the classic Maslach burnout profile: exhaustion, disengagement, and reduced efficacy."
        ))
        
        # R059: Emotional exhaustion burnout
        self.rules.append(Rule(
            rule_id="R059",
            name="Emotional Exhaustion Pattern",
            description="High overwhelm + low mood + high work hours = emotional burnout",
            category="burnout",
            conditions=lambda f: (
                f.get('overwhelm_feeling', 0) >= 7 and
                f.get('mood_score', 5) <= 3 and
                f.get('work_hours', 8) >= 8,
                0.82
            ),
            conclusion="burnout_risk",
            certainty_factor=0.82,
            priority=10,
            explanation="Emotional exhaustion is the core component of burnout. Feeling overwhelmed (7+), very low mood (3-), and extended work hours together indicate advanced emotional depletion."
        ))
        
        # R060: Burnout from lack of balance
        self.rules.append(Rule(
            rule_id="R060",
            name="Burnout from Imbalance",
            description="Poor work-life balance + no relaxation + high work hours",
            category="burnout",
            conditions=lambda f: (
                f.get('work_life_balance', 5) <= 3 and
                f.get('relaxation_activities', 'sometimes') in ['never', 'rarely'] and
                f.get('work_hours', 8) >= 9,
                0.78
            ),
            conclusion="burnout_risk",
            certainty_factor=0.78,
            priority=10,
            explanation="The combination of poor work-life balance, absence of relaxation activities, and extended work hours creates a burnout recipe - maximum input with no recovery."
        ))
        
        # R061: Burnout increases stress levels
        self.rules.append(Rule(
            rule_id="R061",
            name="Burnout Elevates Overall Stress",
            description="Detected burnout risk increases overall stress assessment",
            category="burnout",
            conditions=lambda f: (
                f.get('burnout_risk', False) == True,
                0.85
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.85,
            priority=10,
            explanation="Burnout is a state of chronic stress that has crossed into exhaustion. Its presence confirms high psychological stress and requires intervention."
        ))
    
    # ============================================================
    # ANXIETY DISORDER RISK RULES (R062-R064)
    # ============================================================
    
    def _add_anxiety_rules(self):
        """Rules for detecting anxiety disorder risk"""
        
        # R062: High anxiety with physical symptoms
        self.rules.append(Rule(
            rule_id="R062",
            name="Anxiety with Physical Manifestation",
            description="High anxiety with physical symptoms suggests anxiety disorder",
            category="anxiety",
            conditions=lambda f: (
                f.get('anxiety_level', 0) >= 7 and
                f.get('heart_rate', 70) > 85 and
                f.get('muscle_tension', 0) >= 6,
                0.80
            ),
            conclusion="anxiety_disorder_risk",
            certainty_factor=0.80,
            priority=10,
            explanation="High anxiety (7+/10) manifesting in physical symptoms (elevated heart rate + muscle tension) suggests anxiety may have reached a clinical level requiring professional evaluation."
        ))
        
        # R063: Worry-based anxiety pattern
        self.rules.append(Rule(
            rule_id="R063",
            name="Chronic Worry Pattern",
            description="Constant worry with concentration issues suggests GAD-like pattern",
            category="anxiety",
            conditions=lambda f: (
                f.get('worry_frequency', 'never') == 'always' and
                f.get('concentration_ability', 5) <= 4 and
                f.get('anxiety_level', 0) >= 6,
                0.78
            ),
            conclusion="anxiety_disorder_risk",
            certainty_factor=0.78,
            priority=9,
            explanation="Constant worry combined with concentration difficulties and high anxiety matches Generalized Anxiety Disorder (GAD) patterns. Professional assessment is recommended."
        ))
        
        # R064: Anxiety-sleep-isolation triangle
        self.rules.append(Rule(
            rule_id="R064",
            name="Anxiety-Sleep-Isolation Triangle",
            description="High anxiety + poor sleep + isolation creates self-reinforcing cycle",
            category="anxiety",
            conditions=lambda f: (
                f.get('anxiety_level', 0) >= 6 and
                f.get('sleep_quality', 5) <= 4 and
                f.get('isolation_feeling', 0) >= 5,
                0.82
            ),
            conclusion="anxiety_disorder_risk",
            certainty_factor=0.82,
            priority=10,
            explanation="High anxiety, poor sleep, and social isolation form a self-reinforcing triangle. Each factor worsens the others, creating an escalating stress cycle that's hard to break alone."
        ))
    
    # ============================================================
    # COMPOSITE/INTEGRATION RULES (R065-R069)
    # ============================================================
    
    def _add_composite_stress_rules(self):
        """Rules that integrate multiple domains"""
        
        # R065: Biopsychosocial stress overload
        self.rules.append(Rule(
            rule_id="R065",
            name="Biopsychosocial Stress Overload",
            description="Stress across all three domains indicates comprehensive overload",
            category="composite",
            conditions=lambda f: (
                f.get('physical_stress_high', False) == True and
                f.get('psychological_stress_high', False) == True and
                f.get('lifestyle_risk_high', False) == True,
                0.95
            ),
            conclusion="burnout_risk",
            certainty_factor=0.95,
            priority=12,
            explanation="Stress indicators across physical, psychological, AND lifestyle domains simultaneously indicates comprehensive stress overload - the highest risk state for burnout and health impacts."
        ))
        
        # R066: Physical-Psychological stress amplification
        self.rules.append(Rule(
            rule_id="R066",
            name="Physical-Psychological Amplification",
            description="Physical and psychological stress reinforce each other",
            category="composite",
            conditions=lambda f: (
                f.get('physical_stress_high', False) == True and
                f.get('psychological_stress_high', False) == True,
                0.15  # Increases both
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.15,
            priority=9,
            explanation="Physical and psychological stress don't just add together - they amplify each other. Physical stress increases psychological stress and vice versa through cortisol and nervous system pathways."
        ))
        
        # R067: Social isolation amplifies stress
        self.rules.append(Rule(
            rule_id="R067",
            name="Social Isolation Amplifies Stress",
            description="Social isolation worsens both physical and psychological stress",
            category="composite",
            conditions=lambda f: (
                f.get('social_isolation_risk', False) == True and
                (f.get('physical_stress_high', False) or 
                 f.get('psychological_stress_high', False)),
                0.20
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.20,
            priority=8,
            explanation="Social isolation removes the buffering effect of social support, amplifying the impact of both physical and psychological stressors."
        ))
        
        # R068: Young adult stress pattern
        self.rules.append(Rule(
            rule_id="R068",
            name="Young Adult Stress Pattern",
            description="Young adults with financial + social stress need attention",
            category="composite",
            conditions=lambda f: (
                f.get('age', 35) < 30 and
                f.get('financial_stress', 0) >= 6 and
                f.get('isolation_feeling', 0) >= 5,
                0.65
            ),
            conclusion="psychological_stress_high",
            certainty_factor=0.65,
            priority=8,
            explanation="Young adults (under 30) facing financial stress and social isolation are at elevated stress risk. This life stage brings unique challenges with fewer established coping resources."
        ))
        
        # R069: Age-related resilience
        self.rules.append(Rule(
            rule_id="R069",
            name="Mature Adult Stress Resilience",
            description="Older adults often have better stress coping mechanisms",
            category="composite",
            conditions=lambda f: (
                f.get('age', 35) > 45 and
                f.get('social_support', 5) >= 6,
                -0.15
            ),
            conclusion="lifestyle_risk_high",
            certainty_factor=-0.15,
            priority=6,
            explanation="Adults over 45 with good social support often demonstrate greater emotional regulation and coping experience, providing some resilience against stress."
        ))
    
    # ============================================================
    # STRESS LEVEL DETERMINATION RULES (R070-R079)
    # ============================================================
    
    def _add_stress_level_rules(self):
        """Final rules that determine overall stress level"""
        
        # R070: Minimal stress
        self.rules.append(Rule(
            rule_id="R070",
            name="Minimal Stress Level",
            description="All indicators suggest minimal stress",
            category="stress_level",
            conditions=lambda f: (
                not f.get('physical_stress_high', False) and
                not f.get('psychological_stress_high', False) and
                not f.get('lifestyle_risk_high', False) and
                f.get('sleep_hours', 8) >= 7 and
                f.get('anxiety_level', 0) <= 2,
                0.85
            ),
            conclusion="stress_level_minimal",
            certainty_factor=0.85,
            priority=5,
            explanation="Your overall assessment shows minimal stress. Your sleep, physical health, psychological wellbeing, and lifestyle all appear to be in good balance."
        ))
        
        # R071: Low stress
        self.rules.append(Rule(
            rule_id="R071",
            name="Low Stress Level",
            description="Minor stress indicators present but generally low",
            category="stress_level",
            conditions=lambda f: (
                (f.get('physical_stress_high', False) != True and
                 f.get('psychological_stress_high', False) != True) and
                (f.get('anxiety_level', 0) <= 4 and
                 f.get('mood_score', 5) >= 6),
                0.75
            ),
            conclusion="stress_level_low",
            certainty_factor=0.75,
            priority=5,
            explanation="Your stress level appears low. While there may be minor stressors present, they are manageable and not significantly impacting your wellbeing."
        ))
        
        # R072: Moderate stress (one domain affected)
        self.rules.append(Rule(
            rule_id="R072",
            name="Moderate Stress - Single Domain",
            description="One stress domain significantly affected",
            category="stress_level",
            conditions=lambda f: (
                (f.get('physical_stress_high', False) == True) !=
                (f.get('psychological_stress_high', False) == True) and
                not f.get('burnout_risk', False),
                0.70
            ),
            conclusion="stress_level_moderate",
            certainty_factor=0.70,
            priority=6,
            explanation="Moderate stress is present in one domain (physical or psychological). This level of stress is manageable with proper attention and stress management strategies."
        ))
        
        # R073: Moderate-high stress (lifestyle risk high)
        self.rules.append(Rule(
            rule_id="R073",
            name="Moderate-High Stress - Lifestyle Risk",
            description="High lifestyle risk with moderate symptoms",
            category="stress_level",
            conditions=lambda f: (
                f.get('lifestyle_risk_high', False) == True and
                (f.get('psychological_stress_high', False) or 
                 f.get('physical_stress_high', False)) and
                not f.get('burnout_risk', False),
                0.72
            ),
            conclusion="stress_level_moderate",
            certainty_factor=0.72,
            priority=6,
            explanation="Your lifestyle factors combined with stress symptoms indicate moderate-to-high stress. Lifestyle modifications could significantly reduce your stress levels."
        ))
        
        # R074: High stress (two domains affected)
        self.rules.append(Rule(
            rule_id="R074",
            name="High Stress - Multiple Domains",
            description="Both physical and psychological stress present",
            category="stress_level",
            conditions=lambda f: (
                f.get('physical_stress_high', False) == True and
                f.get('psychological_stress_high', False) == True and
                not f.get('burnout_risk', False),
                0.82
            ),
            conclusion="stress_level_high",
            certainty_factor=0.82,
            priority=7,
            explanation="High stress is evident in both physical and psychological domains. This level requires active intervention to prevent progression to burnout."
        ))
        
        # R075: High stress with social isolation
        self.rules.append(Rule(
            rule_id="R075",
            name="High Stress with Isolation",
            description="High stress combined with social isolation",
            category="stress_level",
            conditions=lambda f: (
                (f.get('physical_stress_high', False) or 
                 f.get('psychological_stress_high', False)) and
                f.get('social_isolation_risk', False) == True and
                not f.get('burnout_risk', False),
                0.78
            ),
            conclusion="stress_level_high",
            certainty_factor=0.78,
            priority=7,
            explanation="The combination of significant stress and social isolation creates a high-risk situation. Without social support, stress management becomes more challenging."
        ))
        
        # R076: Severe stress (burnout risk present)
        self.rules.append(Rule(
            rule_id="R076",
            name="Severe Stress - Burnout Risk",
            description="Burnout risk with high physical and psychological stress",
            category="stress_level",
            conditions=lambda f: (
                f.get('burnout_risk', False) == True and
                f.get('physical_stress_high', False) == True,
                0.87
            ),
            conclusion="stress_level_severe",
            certainty_factor=0.87,
            priority=8,
            explanation="Severe stress is indicated by burnout risk combined with significant physical symptoms. Immediate intervention is needed to prevent further deterioration."
        ))
        
        # R077: Severe stress (anxiety disorder risk)
        self.rules.append(Rule(
            rule_id="R077",
            name="Severe Stress - Anxiety Risk",
            description="Anxiety disorder risk with multiple symptoms",
            category="stress_level",
            conditions=lambda f: (
                f.get('anxiety_disorder_risk', False) == True and
                f.get('psychological_stress_high', False) == True,
                0.85
            ),
            conclusion="stress_level_severe",
            certainty_factor=0.85,
            priority=8,
            explanation="Severe stress with anxiety disorder risk indicators present. Professional mental health support is strongly recommended."
        ))
        
        # R078: Critical stress (all domains + burnout)
        self.rules.append(Rule(
            rule_id="R078",
            name="Critical Stress Level",
            description="All domains affected with burnout and possible anxiety",
            category="stress_level",
            conditions=lambda f: (
                f.get('burnout_risk', False) == True and
                f.get('physical_stress_high', False) == True and
                f.get('psychological_stress_high', False) == True and
                f.get('lifestyle_risk_high', False) == True,
                0.92
            ),
            conclusion="stress_level_critical",
            certainty_factor=0.92,
            priority=9,
            explanation="CRITICAL stress level detected. All stress domains are significantly affected. Immediate professional support is urgently recommended. Please reach out to a healthcare provider."
        ))
        
        # R079: Critical with extreme anxiety
        self.rules.append(Rule(
            rule_id="R079",
            name="Critical - Anxiety + Burnout + Isolation",
            description="Extreme pattern requiring immediate professional help",
            category="stress_level",
            conditions=lambda f: (
                f.get('anxiety_disorder_risk', False) == True and
                f.get('burnout_risk', False) == True and
                f.get('social_isolation_risk', False) == True,
                0.93
            ),
            conclusion="stress_level_critical",
            certainty_factor=0.93,
            priority=9,
            explanation="CRITICAL: The combination of anxiety disorder risk, burnout, and social isolation represents the most severe stress pattern. Urgent professional intervention is essential."
        ))
    
    # ============================================================
    # PROTECTIVE FACTOR RULES (R080-R083)
    # ============================================================
    
    def _add_protective_factor_rules(self):
        """Rules for protective/resilience factors"""
        
        # R080: High resilience pattern
        self.rules.append(Rule(
            rule_id="R080",
            name="High Resilience Indicators",
            description="Multiple protective factors present",
            category="protective",
            conditions=lambda f: (
                f.get('social_support', 5) >= 7 and
                f.get('exercise_frequency', 0) >= 3 and
                f.get('sleep_hours', 8) >= 7 and
                f.get('relaxation_activities', 'never') not in ['never', 'rarely'],
                -0.40
            ),
            conclusion="psychological_stress_high",
            certainty_factor=-0.40,
            priority=9,
            explanation="Multiple protective factors are present: good social support, regular exercise, adequate sleep, and relaxation practices. These significantly buffer against stress impacts."
        ))
        
        # R081: Good mood as protective factor
        self.rules.append(Rule(
            rule_id="R081",
            name="Positive Mood Protection",
            description="High mood score provides psychological resilience",
            category="protective",
            conditions=lambda f: (
                f.get('mood_score', 5) >= 8,
                -0.30
            ),
            conclusion="psychological_stress_high",
            certainty_factor=-0.30,
            priority=7,
            explanation="A high mood score (8+/10) indicates psychological resilience and positive emotional resources that buffer against stress impacts."
        ))
        
        # R082: Low anxiety protective
        self.rules.append(Rule(
            rule_id="R082",
            name="Low Anxiety Resilience",
            description="Very low anxiety indicates strong stress resilience",
            category="protective",
            conditions=lambda f: (
                f.get('anxiety_level', 0) <= 2 and
                f.get('worry_frequency', 'often') in ['never', 'rarely'],
                -0.35
            ),
            conclusion="psychological_stress_high",
            certainty_factor=-0.35,
            priority=7,
            explanation="Very low anxiety and minimal worrying indicate strong psychological resilience and effective coping mechanisms."
        ))
        
        # R083: Work-life balance protection
        self.rules.append(Rule(
            rule_id="R083",
            name="Good Work-Life Balance",
            description="Good work-life balance protects against burnout",
            category="protective",
            conditions=lambda f: (
                f.get('work_life_balance', 5) >= 7 and
                f.get('work_hours', 8) <= 8,
                -0.35
            ),
            conclusion="burnout_risk",
            certainty_factor=-0.35,
            priority=8,
            explanation="Good work-life balance (7+/10) combined with reasonable work hours provides strong protection against burnout. Sustainable work patterns preserve long-term wellbeing."
        ))
    
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def get_rules_by_category(self, category: str) -> List[Rule]:
        """Get all rules in a specific category"""
        return [r for r in self.rules if r.category == category]
    
    def get_rule_by_id(self, rule_id: str) -> Optional[Rule]:
        """Get a specific rule by ID"""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None
    
    def get_rule_count(self) -> int:
        """Get total number of rules"""
        return len(self.rules)
    
    def get_categories(self) -> List[str]:
        """Get all unique categories"""
        return list(set(r.category for r in self.rules))
    
    def get_rules_for_conclusion(self, conclusion: str) -> List[Rule]:
        """Get all rules that produce a specific conclusion"""
        return [r for r in self.rules if r.conclusion == conclusion]
    
    def get_summary(self) -> Dict:
        """Get summary statistics of the rule base"""
        categories = {}
        for rule in self.rules:
            categories[rule.category] = categories.get(rule.category, 0) + 1
        
        return {
            'total_rules': len(self.rules),
            'categories': categories,
            'unique_conclusions': len(set(r.conclusion for r in self.rules)),
            'avg_certainty': sum(abs(r.certainty_factor) for r in self.rules) / len(self.rules)
        }