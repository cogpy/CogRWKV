"""
OpenCog Integration Module for ChatRWKV
Provides AtomSpace-inspired knowledge representation and autonomous agent capabilities.
"""

from .atomspace import AtomSpace, Atom, Node, Link, TruthValue, AtomType
from .pattern_matcher import PatternMatcher, Query, Pattern
from .agent import AutonomousAgent, Goal, Action, GoalStatus
from .cognitive_architecture import CognitiveArchitecture
from .rwkv_integration import RWKVCognitiveEngine

__version__ = "1.0.0"
__all__ = [
    "AtomSpace",
    "Atom", 
    "Node",
    "Link",
    "TruthValue",
    "AtomType",
    "PatternMatcher",
    "Query", 
    "Pattern",
    "AutonomousAgent",
    "Goal",
    "Action",
    "GoalStatus",
    "CognitiveArchitecture",
    "RWKVCognitiveEngine"
]