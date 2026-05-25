"""
Knowledge Base Package
Contains facts (ontology) and rules for StressSense AI
"""

from .facts import Fact, FactBase
from .rules import RuleBase

__all__ = ['Fact', 'FactBase', 'RuleBase']