"""
AUTOMATED TEST SUITE
Mental Stress Screening System

Runs 8 predefined test cases and reports pass/fail results.
Each test case has:
  - A description of the patient profile
  - Input data (patient symptoms and lifestyle)
  - Expected stress level (one or more acceptable values)
  - Expected score range (min, max)

Usage:
    python testing/test_scenarios.py
"""

import sys
import os

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.engine import StressScreeningEngine


# ============================================================
# DEFAULT DATA (used as base for all test cases)
# ============================================================

BASE_DATA = {
    "age": 25, "gender": "unknown", "occupation": "unknown",
    "headache": "none", "fatigue": "none", "chest_pain": "no",
    "muscle_tension": "none", "rapid_heartbeat": "no",
    "stomach_issues": "none", "sweating": "normal",
    "sleep_quality": "good", "appetite": "normal",
    "dizziness": "no", "skin_problems": "no",
    "anxiety": "none", "depression_signs": "no",
    "irritability": "none", "concentration": "good",
    "memory": "good", "mood_swings": "none",
    "overwhelmed": "no", "negative_thoughts": "none",
    "panic_attacks": "no", "hopelessness": "no",
    "self_harm_thoughts": "no", "low_self_esteem": "no",
    "sleep_hours": 8, "social_withdrawal": "none",
    "substance_use": "normal", "work_performance": "normal",
    "procrastination": "mild", "aggression": "normal",
    "crying_spells": "none", "hobbies": "active",
    "screen_time": "normal", "work_hours": 8,
    "exercise": "regular", "diet_quality": "good",
    "social_support": "strong", "coping_skills": "good",
    "work_pressure": "low", "financial_stress": "none",
    "relationship_issues": "none", "academic_pressure": "none",
    "health_concerns": "none", "family_issues": "none",
    "traumatic_event": "no"
}


# ============================================================
# TEST CASE DEFINITIONS
# ============================================================

TEST_CASES = [
    {
        "id": "TC1",
        "name": "Healthy Individual",
        "description": (
            "22-year-old student with regular exercise, "
            "good sleep, strong social support and no significant symptoms."
        ),
        "data": {
            "age": 22, "occupation": "student",
            "headache": "occasional", "fatigue": "mild",
            "sleep_hours": 7, "anxiety": "mild",
            "academic_pressure": "moderate",
            "exercise": "regular", "social_support": "strong",
            "concentration": "good", "coping_skills": "good"
        },
        "expected_levels": ["Minimal", "Mild"],
        "expected_score_range": (0, 35)
    },
    {
        "id": "TC2",
        "name": "Mild Work Stress",
        "description": (
            "30-year-old employee with occasional headaches, "
            "mild anxiety, 6 hours sleep and moderate work pressure."
        ),
        "data": {
            "age": 30, "occupation": "employed",
            "headache": "occasional", "fatigue": "mild",
            "sleep_hours": 6, "anxiety": "mild",
            "work_pressure": "moderate", "exercise": "occasional",
            "social_support": "moderate"
        },
        "expected_levels": ["Mild"],
        "expected_score_range": (10, 40)
    },
    {
        "id": "TC3",
        "name": "Moderate Work Pressure",
        "description": (
            "35-year-old with frequent headaches, moderate anxiety, "
            "poor concentration, feeling overwhelmed and "
            "10+ hour workdays."
        ),
        "data": {
            "age": 35, "occupation": "employed",
            "headache": "frequent", "fatigue": "moderate",
            "sleep_hours": 6, "anxiety": "moderate",
            "work_pressure": "high", "work_hours": 11,
            "concentration": "fair", "overwhelmed": "yes",
            "exercise": "none", "social_support": "moderate"
        },
        "expected_levels": ["Moderate", "Severe"],
        "expected_score_range": (40, 75)
    },
    {
        "id": "TC4",
        "name": "Classic Burnout Pattern",
        "description": (
            "40-year-old professional working 14 hours/day "
            "with the full burnout triad: overwhelmed, "
            "social withdrawal, and severe fatigue."
        ),
        "data": {
            "age": 40,
            "fatigue": "severe", "overwhelmed": "yes",
            "social_withdrawal": "significant",
            "work_hours": 14, "exercise": "none",
            "work_pressure": "high", "social_support": "poor",
            "sleep_hours": 5, "work_performance": "declined",
            "hobbies": "abandoned", "irritability": "frequent",
            "anxiety": "moderate", "coping_skills": "poor"
        },
        "expected_levels": ["Severe", "Critical"],
        "expected_score_range": (60, 100)
    },
    {
        "id": "TC5",
        "name": "Critical Depression Risk",
        "description": (
            "28-year-old with depression signs, hopelessness, "
            "panic attacks, only 3 hours sleep, "
            "increased substance use and severe financial stress."
        ),
        "data": {
            "age": 28,
            "depression_signs": "yes", "hopelessness": "yes",
            "anxiety": "severe", "fatigue": "severe",
            "panic_attacks": "yes", "sleep_hours": 3,
            "social_withdrawal": "significant",
            "substance_use": "increased",
            "hobbies": "abandoned",
            "work_performance": "declined",
            "overwhelmed": "yes",
            "financial_stress": "severe",
            "relationship_issues": "severe",
            "social_support": "none"
        },
        "expected_levels": ["Critical"],
        "expected_score_range": (75, 100)
    },
    {
        "id": "TC6",
        "name": "Protected Individual",
        "description": (
            "32-year-old with frequent headaches and mild anxiety "
            "but protected by regular exercise, "
            "strong social support and excellent coping skills."
        ),
        "data": {
            "age": 32,
            "headache": "frequent", "anxiety": "mild",
            "work_pressure": "moderate",
            "exercise": "regular",
            "social_support": "strong",
            "coping_skills": "excellent"
        },
        "expected_levels": ["Minimal", "Mild"],
        "expected_score_range": (0, 35)
    },
    {
        "id": "TC7",
        "name": "Multiple Physical Symptoms",
        "description": (
            "27-year-old with multiple physical symptoms: "
            "frequent headaches, severe fatigue, rapid heartbeat, "
            "excessive sweating, frequent stomach issues."
        ),
        "data": {
            "age": 27,
            "headache": "frequent", "fatigue": "severe",
            "rapid_heartbeat": "yes", "sweating": "excessive",
            "stomach_issues": "frequent",
            "muscle_tension": "frequent",
            "sleep_quality": "poor", "appetite": "decreased"
        },
        "expected_levels": ["Moderate", "Severe"],
        "expected_score_range": (45, 85)
    },
    {
        "id": "TC8",
        "name": "Student Academic Stress",
        "description": (
            "20-year-old student with high academic pressure, "
            "5 hours sleep, moderate anxiety, "
            "severe procrastination and poor concentration."
        ),
        "data": {
            "age": 20, "occupation": "student",
            "academic_pressure": "high",
            "sleep_hours": 5, "anxiety": "moderate",
            "procrastination": "severe",
            "concentration": "poor", "memory": "poor",
            "social_support": "moderate",
            "work_hours": 8, "overwhelmed": "yes"
        },
        "expected_levels": ["Moderate", "Severe"],
        "expected_score_range": (35, 75)
    }
]


# ============================================================
# TEST RUNNER
# ============================================================

class SystemTester:
    """Runs all test cases and generates a formatted report."""

    def __init__(self):
        self.engine = StressScreeningEngine()
        self.results = []

    def run_single_test(self, test_case):
        """Run one test case and return result dict."""
        test_data = {**BASE_DATA, **test_case["data"]}
        result = self.engine.run_screening(test_data)

        actual_level = result["stress_level"]["level"]
        actual_score = result["stress_score"]
        expected_levels = test_case["expected_levels"]
        score_min, score_max = test_case["expected_score_range"]

        level_pass = actual_level in expected_levels
        score_pass = score_min <= actual_score <= score_max
        overall_pass = level_pass and score_pass

        return {
            "id": test_case["id"],
            "name": test_case["name"],
            "pass": overall_pass,
            "level_pass": level_pass,
            "score_pass": score_pass,
            "expected_levels": expected_levels,
            "actual_level": actual_level,
            "expected_score_range": (score_min, score_max),
            "actual_score": actual_score,
            "rules_fired": result["rules_count"],
            "certainty": result["certainty"],
            "risk_factors": len(result["risk_factors"])
        }

    def run_all_tests(self):
        """Run all test cases and print the full report."""
        self.results = []

        print()
        print("=" * 70)
        print("   MENTAL STRESS SCREENING SYSTEM — AUTOMATED TEST REPORT")
        print("=" * 70)
        print()

        passed_count = 0
        failed_count = 0

        for tc in TEST_CASES:
            result = self.run_single_test(tc)
            self.results.append(result)

            status = "✅ PASS" if result["pass"] else "❌ FAIL"
            if result["pass"]:
                passed_count += 1
            else:
                failed_count += 1

            print(f"{status}  |  {result['id']}: {result['name']}")
            print(f"         Expected Level : {result['expected_levels']}")
            print(f"         Actual Level   : {result['actual_level']}")
            print(f"         Expected Score : {result['expected_score_range']}")
            print(f"         Actual Score   : {result['actual_score']}")
            print(f"         Rules Fired    : {result['rules_fired']}")
            print(f"         Certainty      : {result['certainty']:.2f}")
            print(f"         Risk Factors   : {result['risk_factors']}")

            if not result["level_pass"]:
                print(
                    f"         ⚠️  Level mismatch: got '{result['actual_level']}' "
                    f"expected {result['expected_levels']}"
                )
            if not result["score_pass"]:
                print(
                    f"         ⚠️  Score out of range: {result['actual_score']} "
                    f"not in {result['expected_score_range']}"
                )
            print()

        total = passed_count + failed_count
        pass_rate = (passed_count / total * 100) if total > 0 else 0

        print("=" * 70)
        print(f"   TEST SUMMARY")
        print(f"   Total   : {total}")
        print(f"   Passed  : {passed_count}")
        print(f"   Failed  : {failed_count}")
        print(f"   Pass Rate: {pass_rate:.1f}%")
        print("=" * 70)
        print()

        return self.results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    tester = SystemTester()
    tester.run_all_tests()