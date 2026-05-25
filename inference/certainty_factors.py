"""
Certainty Factor Calculator for StressSense AI
Implements MYCIN-style certainty factor combination
"""

from typing import List, Tuple, Dict, Optional


class CertaintyFactorCalculator:
    """
    Implements certainty factor arithmetic following MYCIN methodology
    
    Certainty Factors represent the net belief in a hypothesis:
    - CF = 1.0: Definitely true
    - CF = 0.0: Unknown/no evidence
    - CF = -1.0: Definitely false
    
    The system combines evidence from multiple rules to build
    a composite certainty for each conclusion.
    """
    
    @staticmethod
    def combine_sequential(cf1: float, cf2: float) -> float:
        """
        Combine certainty factors sequentially (AND combination)
        Used when both conditions must be true
        
        CF(P and Q) = min(CF(P), CF(Q)) for positive CFs
        
        Args:
            cf1: First certainty factor
            cf2: Second certainty factor
        
        Returns:
            Combined certainty factor
        """
        if cf1 >= 0 and cf2 >= 0:
            return min(cf1, cf2)
        elif cf1 < 0 and cf2 < 0:
            return max(cf1, cf2)
        else:
            # Mixed signs - take the smaller absolute value
            return cf1 + cf2 * (1 - abs(cf1))
    
    @staticmethod
    def combine_parallel(cf1: float, cf2: float) -> float:
        """
        Combine certainty factors in parallel (OR/accumulation combination)
        Used when either condition can support the conclusion
        
        MYCIN Formula:
        - Both positive: CF = CF1 + CF2 * (1 - CF1)
        - Both negative: CF = CF1 + CF2 * (1 + CF1)
        - Mixed signs: CF = (CF1 + CF2) / (1 - min(|CF1|, |CF2|))
        
        Args:
            cf1: First certainty factor
            cf2: Second certainty factor
        
        Returns:
            Combined certainty factor
        """
        if cf1 >= 0 and cf2 >= 0:
            return cf1 + cf2 * (1 - cf1)
        elif cf1 <= 0 and cf2 <= 0:
            return cf1 + cf2 * (1 + cf1)
        else:
            denominator = 1 - min(abs(cf1), abs(cf2))
            if abs(denominator) < 1e-10:  # Avoid division by zero
                return 0.0
            return (cf1 + cf2) / denominator
    
    @staticmethod
    def combine_with_rule_cf(evidence_cf: float, rule_cf: float) -> float:
        """
        Combine evidence certainty with rule certainty
        
        MYCIN Formula: CF(conclusion) = CF(evidence) * CF(rule)
        
        This accounts for both the certainty of the evidence
        and the certainty built into the rule itself.
        
        Args:
            evidence_cf: How certain we are about the evidence
            rule_cf: How certain the rule is (from expert)
        
        Returns:
            Net certainty for the conclusion
        """
        return evidence_cf * rule_cf
    
    @classmethod
    def accumulate_evidence(cls, existing_cf: float, new_cf: float) -> float:
        """
        Accumulate new evidence with existing certainty
        Uses parallel combination (evidence accumulates)
        
        Args:
            existing_cf: Current certainty for hypothesis
            new_cf: New evidence certainty
        
        Returns:
            Updated certainty factor
        """
        return cls.combine_parallel(existing_cf, new_cf)
    
    @classmethod
    def combine_multiple(cls, cf_list: List[float]) -> float:
        """
        Combine multiple certainty factors
        Applies parallel combination iteratively
        
        Args:
            cf_list: List of certainty factors to combine
        
        Returns:
            Combined certainty factor
        """
        if not cf_list:
            return 0.0
        
        result = cf_list[0]
        for cf in cf_list[1:]:
            result = cls.combine_parallel(result, cf)
        
        return result
    
    @staticmethod
    def negate(cf: float) -> float:
        """
        Negate a certainty factor
        CF(NOT A) = -CF(A)
        
        Args:
            cf: Certainty factor to negate
        
        Returns:
            Negated certainty factor
        """
        return -cf
    
    @staticmethod
    def normalize(cf: float) -> float:
        """
        Normalize certainty factor to [-1, 1] range
        
        Args:
            cf: Certainty factor to normalize
        
        Returns:
            Normalized certainty factor
        """
        return max(-1.0, min(1.0, cf))
    
    @staticmethod
    def cf_to_probability(cf: float) -> float:
        """
        Convert certainty factor to probability estimate
        
        Mapping:
        - CF = 1.0 → P = 1.0
        - CF = 0.0 → P = 0.5 (unknown)
        - CF = -1.0 → P = 0.0
        
        Args:
            cf: Certainty factor
        
        Returns:
            Probability estimate [0, 1]
        """
        return (cf + 1.0) / 2.0
    
    @staticmethod
    def probability_to_cf(probability: float) -> float:
        """
        Convert probability to certainty factor
        
        Args:
            probability: Probability value [0, 1]
        
        Returns:
            Certainty factor [-1, 1]
        """
        return 2 * probability - 1.0
    
    @classmethod
    def calculate_net_stress_cf(cls, 
                                 stress_evidence: Dict[str, float]) -> float:
        """
        Calculate net certainty factor for overall stress assessment
        
        Combines evidence from different stress domains with appropriate weights:
        - Physical stress: 25%
        - Psychological stress: 30%
        - Lifestyle risk: 20%
        - Social isolation: 15%
        - Burnout risk: 10%
        
        Args:
            stress_evidence: Dictionary of {domain: certainty_factor}
        
        Returns:
            Net stress certainty factor
        """
        weights = {
            'physical_stress': 0.25,
            'psychological_stress': 0.30,
            'lifestyle_risk': 0.20,
            'social_isolation': 0.15,
            'burnout_risk': 0.10
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for domain, weight in weights.items():
            if domain in stress_evidence:
                weighted_sum += stress_evidence[domain] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return cls.normalize(weighted_sum / total_weight)
    
    @staticmethod
    def cf_to_stress_score(cf: float) -> float:
        """
        Convert certainty factor to 0-100 stress score
        
        Args:
            cf: Certainty factor [-1, 1]
        
        Returns:
            Stress score [0, 100]
        """
        # CF of -1 = minimal stress (score ~5)
        # CF of 0 = moderate stress (score ~50)
        # CF of 1 = maximum stress (score ~95)
        normalized = (cf + 1.0) / 2.0  # Map to [0, 1]
        return 5 + normalized * 90  # Map to [5, 95]
    
    @classmethod
    def calculate_domain_score(cls, 
                                domain_cfs: List[Tuple[str, float]]) -> float:
        """
        Calculate aggregate certainty for a domain
        
        Args:
            domain_cfs: List of (rule_id, certainty_factor) tuples
        
        Returns:
            Aggregate certainty factor for domain
        """
        if not domain_cfs:
            return 0.0
        
        cf_values = [cf for _, cf in domain_cfs]
        return cls.combine_multiple(cf_values)
    
    @staticmethod
    def interpret_cf(cf: float) -> str:
        """
        Provide human-readable interpretation of certainty factor
        
        Args:
            cf: Certainty factor value
        
        Returns:
            Human-readable interpretation
        """
        if cf >= 0.8:
            return "Almost certainly true (very high confidence)"
        elif cf >= 0.6:
            return "Probably true (high confidence)"
        elif cf >= 0.4:
            return "Likely true (moderate confidence)"
        elif cf >= 0.2:
            return "Possibly true (low confidence)"
        elif cf > 0:
            return "Slightly possible (very low confidence)"
        elif cf == 0:
            return "Unknown (no evidence)"
        elif cf > -0.2:
            return "Slightly unlikely"
        elif cf > -0.4:
            return "Possibly false (low confidence)"
        elif cf > -0.6:
            return "Probably false (moderate confidence)"
        elif cf > -0.8:
            return "Unlikely true (high confidence)"
        else:
            return "Almost certainly false (very high confidence)"
    
    @staticmethod
    def get_confidence_percentage(cf: float) -> float:
        """
        Convert CF to percentage confidence
        
        Args:
            cf: Certainty factor
        
        Returns:
            Confidence percentage [0, 100]
        """
        return abs(cf) * 100
    
    @classmethod
    def threshold_check(cls, cf: float, threshold: float = 0.2) -> bool:
        """
        Check if certainty factor exceeds threshold for conclusion
        
        Args:
            cf: Certainty factor
            threshold: Minimum CF to consider conclusion supported
        
        Returns:
            True if CF exceeds threshold
        """
        return cf > threshold


class CFTracker:
    """
    Tracks certainty factor changes throughout inference
    Provides detailed audit trail of reasoning
    """
    
    def __init__(self):
        self.updates: List[Dict] = []
        self.final_values: Dict[str, float] = {}
    
    def record_update(self, hypothesis: str, rule_id: str, 
                      old_cf: float, new_cf: float, 
                      rule_cf: float, evidence_cf: float = 1.0):
        """Record a CF update"""
        self.updates.append({
            'hypothesis': hypothesis,
            'rule_id': rule_id,
            'old_cf': old_cf,
            'new_cf': new_cf,
            'rule_cf': rule_cf,
            'evidence_cf': evidence_cf,
            'delta': new_cf - old_cf
        })
        self.final_values[hypothesis] = new_cf
    
    def get_updates_for(self, hypothesis: str) -> List[Dict]:
        """Get all updates for a specific hypothesis"""
        return [u for u in self.updates if u['hypothesis'] == hypothesis]
    
    def get_final_cf(self, hypothesis: str) -> float:
        """Get final certainty factor for hypothesis"""
        return self.final_values.get(hypothesis, 0.0)
    
    def get_strongest_evidence(self, hypothesis: str) -> Optional[Dict]:
        """Get the strongest piece of evidence for a hypothesis"""
        updates = self.get_updates_for(hypothesis)
        if not updates:
            return None
        return max(updates, key=lambda u: abs(u['rule_cf']))
    
    def get_summary(self) -> Dict:
        """Get summary of all CF tracking"""
        return {
            'total_updates': len(self.updates),
            'hypotheses_updated': len(self.final_values),
            'final_values': self.final_values.copy()
        }
    
    def clear(self):
        """Clear all tracking data"""
        self.updates = []
        self.final_values = {}