"""
Fuzzy Logic Module for StressSense AI
Implements fuzzy membership functions for continuous variables
"""

import numpy as np
from typing import Dict, Tuple


class FuzzyMembership:
    """
    Fuzzy Logic Membership Functions
    Converts crisp values into fuzzy set memberships
    """
    
    @staticmethod
    def triangular(x: float, a: float, b: float, c: float) -> float:
        """
        Triangular membership function
        Peak at b, zero at a and c
        
        Args:
            x: Input value
            a: Left boundary (membership = 0)
            b: Peak (membership = 1)
            c: Right boundary (membership = 0)
        """
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)
    
    @staticmethod
    def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
        """
        Trapezoidal membership function
        Full membership between b and c
        
        Args:
            x: Input value
            a: Left lower boundary
            b: Left upper boundary
            c: Right upper boundary
            d: Right lower boundary
        """
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c)
    
    @staticmethod
    def gaussian(x: float, center: float, sigma: float) -> float:
        """
        Gaussian membership function
        Smooth bell curve centered at 'center'
        
        Args:
            x: Input value
            center: Center of the bell curve
            sigma: Width parameter
        """
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    @staticmethod
    def sigmoid(x: float, center: float, slope: float) -> float:
        """
        Sigmoid membership function
        S-shaped curve useful for increasing membership
        
        Args:
            x: Input value
            center: Inflection point
            slope: Steepness (positive = rising, negative = falling)
        """
        return 1.0 / (1.0 + np.exp(-slope * (x - center)))
    
    @staticmethod
    def reverse_sigmoid(x: float, center: float, slope: float) -> float:
        """Decreasing sigmoid function"""
        return 1.0 / (1.0 + np.exp(slope * (x - center)))
    
    # ============================================================
    # STRESS-SPECIFIC MEMBERSHIP FUNCTIONS
    # ============================================================
    
    @classmethod
    def sleep_stress_membership(cls, sleep_hours: float) -> Dict[str, float]:
        """
        Calculate fuzzy memberships for sleep hours
        Returns membership in each sleep quality category
        """
        return {
            'severe_deprivation': cls.trapezoidal(sleep_hours, 0, 0, 4, 5.5),
            'moderate_deprivation': cls.triangular(sleep_hours, 4.5, 6, 7),
            'borderline': cls.triangular(sleep_hours, 6, 7, 7.5),
            'adequate': cls.trapezoidal(sleep_hours, 7, 7.5, 8.5, 9),
            'optimal': cls.trapezoidal(sleep_hours, 7.5, 8, 8.5, 9.5),
            'excessive': cls.trapezoidal(sleep_hours, 9, 10, 24, 24)
        }
    
    @classmethod
    def anxiety_membership(cls, anxiety_level: float) -> Dict[str, float]:
        """
        Calculate fuzzy memberships for anxiety level (0-10 scale)
        """
        return {
            'minimal': cls.trapezoidal(anxiety_level, 0, 0, 1.5, 3),
            'mild': cls.triangular(anxiety_level, 2, 3.5, 5),
            'moderate': cls.triangular(anxiety_level, 4, 5.5, 7),
            'high': cls.triangular(anxiety_level, 6, 7.5, 9),
            'severe': cls.trapezoidal(anxiety_level, 8, 9, 10, 10)
        }
    
    @classmethod
    def stress_level_membership(cls, stress_score: float) -> Dict[str, float]:
        """
        Calculate fuzzy memberships for overall stress score (0-100)
        Returns membership in each stress level category
        """
        return {
            'minimal': cls.trapezoidal(stress_score, 0, 0, 10, 20),
            'low': cls.triangular(stress_score, 10, 25, 40),
            'moderate': cls.triangular(stress_score, 30, 45, 60),
            'high': cls.triangular(stress_score, 50, 65, 80),
            'severe': cls.triangular(stress_score, 70, 82, 92),
            'critical': cls.trapezoidal(stress_score, 85, 92, 100, 100)
        }
    
    @classmethod
    def fatigue_membership(cls, fatigue: float) -> Dict[str, float]:
        """Membership functions for fatigue level"""
        return {
            'energetic': cls.trapezoidal(fatigue, 0, 0, 2, 4),
            'mild_fatigue': cls.triangular(fatigue, 2, 4, 6),
            'moderate_fatigue': cls.triangular(fatigue, 4, 6, 8),
            'severe_fatigue': cls.trapezoidal(fatigue, 7, 8.5, 10, 10)
        }
    
    @classmethod
    def mood_membership(cls, mood: float) -> Dict[str, float]:
        """Membership functions for mood score"""
        return {
            'very_negative': cls.trapezoidal(mood, 0, 0, 2, 3.5),
            'negative': cls.triangular(mood, 2.5, 4, 5.5),
            'neutral': cls.triangular(mood, 4, 5.5, 7),
            'positive': cls.triangular(mood, 5.5, 7, 8.5),
            'very_positive': cls.trapezoidal(mood, 7.5, 9, 10, 10)
        }
    
    @classmethod
    def heart_rate_membership(cls, hr: float) -> Dict[str, float]:
        """Membership functions for resting heart rate"""
        return {
            'low': cls.trapezoidal(hr, 0, 0, 55, 65),
            'normal': cls.trapezoidal(hr, 58, 65, 80, 88),
            'elevated': cls.triangular(hr, 80, 90, 100),
            'high': cls.trapezoidal(hr, 95, 105, 200, 200)
        }
    
    @classmethod
    def work_hours_membership(cls, hours: float) -> Dict[str, float]:
        """Membership functions for daily work hours"""
        return {
            'minimal': cls.trapezoidal(hours, 0, 0, 4, 6),
            'part_time': cls.triangular(hours, 4, 6, 8),
            'full_time': cls.triangular(hours, 7, 8, 9),
            'extended': cls.triangular(hours, 8.5, 10, 12),
            'excessive': cls.trapezoidal(hours, 11, 13, 24, 24)
        }
    
    # ============================================================
    # DEFUZZIFICATION METHODS
    # ============================================================
    
    @staticmethod
    def centroid_defuzzify(memberships: Dict[str, float], 
                           reference_values: Dict[str, float]) -> float:
        """
        Centroid method defuzzification
        Calculates weighted average of reference values
        
        Args:
            memberships: Dictionary of {category: membership_degree}
            reference_values: Dictionary of {category: representative_value}
        """
        numerator = sum(
            memberships.get(cat, 0) * val 
            for cat, val in reference_values.items()
        )
        denominator = sum(memberships.values())
        
        if denominator == 0:
            return 0.0
        return numerator / denominator
    
    @staticmethod
    def max_membership(memberships: Dict[str, float]) -> Tuple[str, float]:
        """
        Find the category with maximum membership
        Returns (category_name, membership_value)
        """
        if not memberships:
            return ("unknown", 0.0)
        max_cat = max(memberships, key=memberships.get)
        return (max_cat, memberships[max_cat])
    
    @classmethod
    def get_stress_category(cls, stress_score: float) -> Tuple[str, Dict[str, float]]:
        """
        Determine stress category from numerical score using fuzzy logic
        Returns (primary_category, all_memberships)
        """
        memberships = cls.stress_level_membership(stress_score)
        primary, _ = cls.max_membership(memberships)
        return primary, memberships
    
    @classmethod
    def fuzzy_stress_score(cls, user_data: Dict) -> float:
        """
        Calculate fuzzy stress score from user data
        Integrates multiple fuzzy assessments
        
        Returns score 0-100
        """
        components = []
        weights = []
        
        # Sleep contribution (weight: 0.25)
        if 'sleep_hours' in user_data:
            sleep_membership = cls.sleep_stress_membership(user_data['sleep_hours'])
            sleep_references = {
                'severe_deprivation': 90,
                'moderate_deprivation': 65,
                'borderline': 45,
                'adequate': 20,
                'optimal': 10,
                'excessive': 40
            }
            sleep_score = cls.centroid_defuzzify(sleep_membership, sleep_references)
            components.append(sleep_score)
            weights.append(0.25)
        
        # Anxiety contribution (weight: 0.25)
        if 'anxiety_level' in user_data:
            anxiety_membership = cls.anxiety_membership(user_data['anxiety_level'])
            anxiety_references = {
                'minimal': 10,
                'mild': 30,
                'moderate': 55,
                'high': 75,
                'severe': 92
            }
            anxiety_score = cls.centroid_defuzzify(anxiety_membership, anxiety_references)
            components.append(anxiety_score)
            weights.append(0.25)
        
        # Fatigue contribution (weight: 0.20)
        if 'fatigue_level' in user_data:
            fatigue_membership = cls.fatigue_membership(user_data['fatigue_level'])
            fatigue_references = {
                'energetic': 10,
                'mild_fatigue': 35,
                'moderate_fatigue': 60,
                'severe_fatigue': 88
            }
            fatigue_score = cls.centroid_defuzzify(fatigue_membership, fatigue_references)
            components.append(fatigue_score)
            weights.append(0.20)
        
        # Mood contribution (weight: 0.15) - inverted (high mood = low stress)
        if 'mood_score' in user_data:
            mood_membership = cls.mood_membership(user_data['mood_score'])
            mood_references = {
                'very_negative': 85,
                'negative': 65,
                'neutral': 45,
                'positive': 25,
                'very_positive': 10
            }
            mood_score = cls.centroid_defuzzify(mood_membership, mood_references)
            components.append(mood_score)
            weights.append(0.15)
        
        # Work hours contribution (weight: 0.15)
        if 'work_hours' in user_data:
            work_membership = cls.work_hours_membership(user_data['work_hours'])
            work_references = {
                'minimal': 15,
                'part_time': 20,
                'full_time': 35,
                'extended': 65,
                'excessive': 85
            }
            work_score = cls.centroid_defuzzify(work_membership, work_references)
            components.append(work_score)
            weights.append(0.15)
        
        # Calculate weighted average
        if not components:
            return 50.0
        
        total_weight = sum(weights)
        if total_weight == 0:
            return 50.0
        
        fuzzy_score = sum(c * w for c, w in zip(components, weights)) / total_weight
        return max(0.0, min(100.0, fuzzy_score))


class FuzzyRuleEvaluator:
    """
    Evaluates conditions using fuzzy logic for soft boundaries
    Allows rules to fire with partial membership
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Minimum membership degree to consider condition met
        """
        self.threshold = threshold
        self.fuzzy = FuzzyMembership()
    
    def evaluate_sleep(self, sleep_hours: float) -> Dict[str, bool]:
        """
        Evaluate sleep-related conditions with fuzzy boundaries
        Returns fuzzy truth values for each condition
        """
        memberships = FuzzyMembership.sleep_stress_membership(sleep_hours)
        return {
            'is_severely_deprived': memberships['severe_deprivation'] > self.threshold,
            'is_moderately_deprived': memberships['moderate_deprivation'] > self.threshold,
            'is_adequate': memberships['adequate'] > self.threshold,
            'is_optimal': memberships['optimal'] > self.threshold,
            'membership': memberships
        }
    
    def evaluate_anxiety(self, anxiety_level: float) -> Dict[str, bool]:
        """Evaluate anxiety with fuzzy logic"""
        memberships = FuzzyMembership.anxiety_membership(anxiety_level)
        return {
            'is_minimal': memberships['minimal'] > self.threshold,
            'is_moderate': memberships['moderate'] > self.threshold,
            'is_high': memberships['high'] > self.threshold,
            'is_severe': memberships['severe'] > self.threshold,
            'membership': memberships
        }
    
    def get_fuzzy_certainty(self, membership_degree: float, 
                            base_cf: float) -> float:
        """
        Adjust certainty factor based on fuzzy membership degree
        Higher membership = higher certainty
        
        Args:
            membership_degree: How much the value belongs to the fuzzy set (0-1)
            base_cf: The rule's base certainty factor
        
        Returns:
            Adjusted certainty factor
        """
        return base_cf * membership_degree