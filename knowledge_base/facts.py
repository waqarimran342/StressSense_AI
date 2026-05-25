"""
Fact Class Definitions - Ontology for StressSense AI
Defines all facts, their types, and relationships
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import time


class FactType(Enum):
    """Types of facts in the knowledge base"""
    SYMPTOM = "symptom"
    LIFESTYLE = "lifestyle"
    PHYSIOLOGICAL = "physiological"
    PSYCHOLOGICAL = "psychological"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    DERIVED = "derived"
    CONCLUSION = "conclusion"


class StressLevel(Enum):
    """Stress level classifications"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    CRITICAL = "critical"


class ConfidenceLevel(Enum):
    """Confidence level classifications"""
    VERY_LOW = "very_low"      # CF: 0.0 - 0.2
    LOW = "low"                 # CF: 0.2 - 0.4
    MODERATE = "moderate"       # CF: 0.4 - 0.6
    HIGH = "high"               # CF: 0.6 - 0.8
    VERY_HIGH = "very_high"    # CF: 0.8 - 1.0


@dataclass
class Fact:
    """
    Core Fact class representing a piece of knowledge
    
    Attributes:
        name: Unique identifier for the fact
        value: The actual value of the fact
        fact_type: Category of the fact
        certainty_factor: Confidence in this fact (0.0 to 1.0)
        source: Where this fact came from
        timestamp: When fact was created
        metadata: Additional information
        derived_from: Which rules/facts derived this fact
    """
    name: str
    value: Any
    fact_type: FactType = FactType.SYMPTOM
    certainty_factor: float = 1.0
    source: str = "user_input"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    derived_from: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate fact after creation"""
        # Clamp certainty factor between -1 and 1
        self.certainty_factor = max(-1.0, min(1.0, self.certainty_factor))
        
        # Validate name
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Fact name must be a non-empty string")
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Get human-readable confidence level"""
        cf = abs(self.certainty_factor)
        if cf < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif cf < 0.4:
            return ConfidenceLevel.LOW
        elif cf < 0.6:
            return ConfidenceLevel.MODERATE
        elif cf < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def is_positive(self) -> bool:
        """Check if fact has positive certainty"""
        return self.certainty_factor > 0
    
    def is_negative(self) -> bool:
        """Check if fact has negative certainty"""
        return self.certainty_factor < 0
    
    def update_certainty(self, new_cf: float):
        """Update certainty factor with validation"""
        self.certainty_factor = max(-1.0, min(1.0, new_cf))
    
    def to_dict(self) -> Dict:
        """Convert fact to dictionary representation"""
        return {
            'name': self.name,
            'value': self.value,
            'fact_type': self.fact_type.value,
            'certainty_factor': self.certainty_factor,
            'source': self.source,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'derived_from': self.derived_from,
            'confidence_level': self.get_confidence_level().value
        }
    
    def __repr__(self) -> str:
        return (f"Fact(name='{self.name}', value={self.value}, "
                f"CF={self.certainty_factor:.2f}, type={self.fact_type.value})")


class FactBase:
    """
    Working Memory - Stores all facts during inference
    
    This is the dynamic knowledge store that holds:
    - User-provided facts
    - Derived facts from rule firing
    - Conclusions reached
    """
    
    def __init__(self):
        self._facts: Dict[str, Fact] = {}
        self._history: List[Dict] = []
        self._inference_trace: List[str] = []
    
    def add_fact(self, fact: Fact) -> None:
        """
        Add a fact to the knowledge base
        If fact already exists, update with combined certainty
        """
        if fact.name in self._facts:
            existing = self._facts[fact.name]
            # Combine certainty factors
            combined_cf = self._combine_certainty(
                existing.certainty_factor, 
                fact.certainty_factor
            )
            existing.update_certainty(combined_cf)
            existing.derived_from.extend(fact.derived_from)
            
            # Log update
            self._history.append({
                'action': 'update',
                'fact_name': fact.name,
                'old_cf': existing.certainty_factor,
                'new_cf': combined_cf,
                'timestamp': time.time()
            })
        else:
            self._facts[fact.name] = fact
            self._history.append({
                'action': 'add',
                'fact_name': fact.name,
                'cf': fact.certainty_factor,
                'timestamp': time.time()
            })
    
    def get_fact(self, name: str) -> Optional[Fact]:
        """Retrieve a fact by name"""
        return self._facts.get(name)
    
    def get_value(self, name: str, default=None) -> Any:
        """Get the value of a fact"""
        fact = self._facts.get(name)
        return fact.value if fact else default
    
    def get_certainty(self, name: str) -> float:
        """Get certainty factor of a fact"""
        fact = self._facts.get(name)
        return fact.certainty_factor if fact else 0.0
    
    def has_fact(self, name: str) -> bool:
        """Check if a fact exists"""
        return name in self._facts
    
    def get_facts_by_type(self, fact_type: FactType) -> List[Fact]:
        """Get all facts of a specific type"""
        return [f for f in self._facts.values() 
                if f.fact_type == fact_type]
    
    def get_all_facts(self) -> Dict[str, Fact]:
        """Get all facts"""
        return self._facts.copy()
    
    def get_derived_facts(self) -> List[Fact]:
        """Get only derived (inferred) facts"""
        return [f for f in self._facts.values() 
                if f.source == "inference"]
    
    def add_to_trace(self, message: str) -> None:
        """Add a step to the inference trace"""
        self._inference_trace.append(message)
    
    def get_trace(self) -> List[str]:
        """Get the inference trace"""
        return self._inference_trace.copy()
    
    def clear_trace(self) -> None:
        """Clear the inference trace"""
        self._inference_trace = []
    
    def get_history(self) -> List[Dict]:
        """Get the fact modification history"""
        return self._history.copy()
    
    def _combine_certainty(self, cf1: float, cf2: float) -> float:
        """
        Combine two certainty factors using MYCIN formula
        For same sign: CF(A,B) = CF(A) + CF(B) * (1 - CF(A))
        For opposite sign: CF(A,B) = CF(A) + CF(B) / (1 - min(|CF(A)|, |CF(B)|))
        """
        if cf1 >= 0 and cf2 >= 0:
            return cf1 + cf2 * (1 - cf1)
        elif cf1 <= 0 and cf2 <= 0:
            return cf1 + cf2 * (1 + cf1)
        else:
            denominator = 1 - min(abs(cf1), abs(cf2))
            if denominator == 0:
                return 0
            return (cf1 + cf2) / denominator
    
    def get_summary(self) -> Dict:
        """Get a summary of the fact base"""
        facts_by_type = {}
        for fact_type in FactType:
            facts = self.get_facts_by_type(fact_type)
            if facts:
                facts_by_type[fact_type.value] = len(facts)
        
        return {
            'total_facts': len(self._facts),
            'facts_by_type': facts_by_type,
            'derived_facts': len(self.get_derived_facts()),
            'trace_steps': len(self._inference_trace),
            'history_entries': len(self._history)
        }
    
    def clear(self) -> None:
        """Clear all facts from the knowledge base"""
        self._facts = {}
        self._history = []
        self._inference_trace = []
    
    def __len__(self) -> int:
        return len(self._facts)
    
    def __contains__(self, name: str) -> bool:
        return name in self._facts
    
    def __repr__(self) -> str:
        return f"FactBase(facts={len(self._facts)}, derived={len(self.get_derived_facts())})"


# ============================================================
# ONTOLOGY DEFINITIONS - Domain Knowledge Structure
# ============================================================

# All possible user input facts with their descriptions
USER_INPUT_FACTS = {
    # ── Physiological Symptoms ──────────────────────────────
    'sleep_hours': {
        'description': 'Average hours of sleep per night',
        'type': FactType.PHYSIOLOGICAL,
        'range': (0, 24),
        'unit': 'hours'
    },
    'sleep_quality': {
        'description': 'Quality of sleep (1-10 scale)',
        'type': FactType.PHYSIOLOGICAL,
        'range': (1, 10),
        'unit': 'scale'
    },
    'heart_rate': {
        'description': 'Resting heart rate',
        'type': FactType.PHYSIOLOGICAL,
        'range': (40, 200),
        'unit': 'bpm'
    },
    'headache_frequency': {
        'description': 'How often headaches occur',
        'type': FactType.PHYSIOLOGICAL,
        'options': ['never', 'rarely', 'sometimes', 'often', 'always']
    },
    'muscle_tension': {
        'description': 'Level of muscle tension/pain',
        'type': FactType.PHYSIOLOGICAL,
        'range': (0, 10),
        'unit': 'scale'
    },
    'appetite_change': {
        'description': 'Change in appetite',
        'type': FactType.PHYSIOLOGICAL,
        'options': ['increased', 'normal', 'decreased', 'severely_decreased']
    },
    'fatigue_level': {
        'description': 'Level of physical fatigue',
        'type': FactType.PHYSIOLOGICAL,
        'range': (0, 10),
        'unit': 'scale'
    },
    'digestive_issues': {
        'description': 'Frequency of digestive problems',
        'type': FactType.PHYSIOLOGICAL,
        'options': ['never', 'rarely', 'sometimes', 'often', 'always']
    },
    
    # ── Psychological Symptoms ──────────────────────────────
    'anxiety_level': {
        'description': 'Level of anxiety experienced',
        'type': FactType.PSYCHOLOGICAL,
        'range': (0, 10),
        'unit': 'scale'
    },
    'mood_score': {
        'description': 'Overall mood (1=very bad, 10=excellent)',
        'type': FactType.PSYCHOLOGICAL,
        'range': (1, 10),
        'unit': 'scale'
    },
    'concentration_ability': {
        'description': 'Ability to concentrate (1-10)',
        'type': FactType.PSYCHOLOGICAL,
        'range': (1, 10),
        'unit': 'scale'
    },
    'irritability_level': {
        'description': 'Level of irritability',
        'type': FactType.PSYCHOLOGICAL,
        'range': (0, 10),
        'unit': 'scale'
    },
    'overwhelm_feeling': {
        'description': 'Feeling overwhelmed',
        'type': FactType.PSYCHOLOGICAL,
        'range': (0, 10),
        'unit': 'scale'
    },
    'motivation_level': {
        'description': 'Level of motivation',
        'type': FactType.PSYCHOLOGICAL,
        'range': (0, 10),
        'unit': 'scale'
    },
    'worry_frequency': {
        'description': 'How often excessive worrying occurs',
        'type': FactType.PSYCHOLOGICAL,
        'options': ['never', 'rarely', 'sometimes', 'often', 'always']
    },
    
    # ── Lifestyle Factors ───────────────────────────────────
    'exercise_frequency': {
        'description': 'Exercise sessions per week',
        'type': FactType.LIFESTYLE,
        'range': (0, 7),
        'unit': 'days/week'
    },
    'work_hours': {
        'description': 'Average work hours per day',
        'type': FactType.LIFESTYLE,
        'range': (0, 24),
        'unit': 'hours/day'
    },
    'caffeine_intake': {
        'description': 'Daily caffeine intake',
        'type': FactType.LIFESTYLE,
        'options': ['none', 'low', 'moderate', 'high', 'very_high']
    },
    'alcohol_consumption': {
        'description': 'Alcohol consumption level',
        'type': FactType.LIFESTYLE,
        'options': ['none', 'occasional', 'moderate', 'frequent', 'heavy']
    },
    'smoking_status': {
        'description': 'Smoking status',
        'type': FactType.LIFESTYLE,
        'options': ['never', 'former', 'occasional', 'regular']
    },
    'relaxation_activities': {
        'description': 'Engagement in relaxation activities',
        'type': FactType.LIFESTYLE,
        'options': ['never', 'rarely', 'sometimes', 'often', 'daily']
    },
    'screen_time_hours': {
        'description': 'Daily screen time (non-work)',
        'type': FactType.LIFESTYLE,
        'range': (0, 16),
        'unit': 'hours'
    },
    
    # ── Environmental Factors ───────────────────────────────
    'work_environment': {
        'description': 'Quality of work environment',
        'type': FactType.ENVIRONMENTAL,
        'options': ['very_positive', 'positive', 'neutral', 'negative', 'very_negative']
    },
    'major_life_changes': {
        'description': 'Number of major life changes in past 6 months',
        'type': FactType.ENVIRONMENTAL,
        'range': (0, 10),
        'unit': 'count'
    },
    'financial_stress': {
        'description': 'Level of financial stress',
        'type': FactType.ENVIRONMENTAL,
        'range': (0, 10),
        'unit': 'scale'
    },
    'living_situation': {
        'description': 'Satisfaction with living situation',
        'type': FactType.ENVIRONMENTAL,
        'options': ['very_satisfied', 'satisfied', 'neutral', 'unsatisfied', 'very_unsatisfied']
    },
    
    # ── Social Factors ──────────────────────────────────────
    'social_support': {
        'description': 'Level of social support available',
        'type': FactType.SOCIAL,
        'range': (0, 10),
        'unit': 'scale'
    },
    'relationship_quality': {
        'description': 'Quality of close relationships',
        'type': FactType.SOCIAL,
        'options': ['excellent', 'good', 'fair', 'poor', 'very_poor']
    },
    'work_life_balance': {
        'description': 'Satisfaction with work-life balance',
        'type': FactType.SOCIAL,
        'range': (1, 10),
        'unit': 'scale'
    },
    'isolation_feeling': {
        'description': 'Feeling of social isolation',
        'type': FactType.SOCIAL,
        'range': (0, 10),
        'unit': 'scale'
    },
    
    # ── Personal Information ─────────────────────────────────
    'age': {
        'description': 'Age of the person',
        'type': FactType.PHYSIOLOGICAL,
        'range': (10, 100),
        'unit': 'years'
    },
    'gender': {
        'description': 'Gender',
        'type': FactType.PHYSIOLOGICAL,
        'options': ['male', 'female', 'other', 'prefer_not_to_say']
    },
    'occupation': {
        'description': 'Current occupation type',
        'type': FactType.LIFESTYLE,
        'options': ['student', 'employed', 'self_employed', 'unemployed', 'retired']
    }
}

# Derived facts that can be inferred
DERIVED_FACT_DEFINITIONS = {
    'poor_sleep': 'Sleep is inadequate in quantity or quality',
    'physical_stress_high': 'Physical stress indicators are elevated',
    'psychological_stress_high': 'Psychological stress indicators are elevated',
    'lifestyle_risk_high': 'Lifestyle factors significantly increase stress risk',
    'social_isolation_risk': 'Social isolation contributes to stress',
    'burnout_risk': 'Risk of burnout or chronic stress',
    'anxiety_disorder_risk': 'Indicators suggesting anxiety-related stress',
    'stress_level_minimal': 'Overall stress level is minimal',
    'stress_level_low': 'Overall stress level is low',
    'stress_level_moderate': 'Overall stress level is moderate',
    'stress_level_high': 'Overall stress level is high',
    'stress_level_severe': 'Overall stress level is severe',
    'stress_level_critical': 'Overall stress level is critical - immediate help needed'
}