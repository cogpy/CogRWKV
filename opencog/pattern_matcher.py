"""
Pattern Matcher - Query and reasoning over AtomSpace
Implements pattern matching capabilities for knowledge retrieval and inference.
"""

from typing import List, Dict, Set, Optional, Any, Callable, Union
from dataclasses import dataclass
import re
from .atomspace import AtomSpace, Atom, AtomType, TruthValue


@dataclass
class QueryResult:
    """Result of a pattern matching query."""
    bindings: Dict[str, Atom]
    score: float = 1.0
    
    def __str__(self):
        bindings_str = ", ".join(f"{var}: {atom.name if atom.name else atom.atom_type}" 
                                for var, atom in self.bindings.items())
        return f"QueryResult({bindings_str}, score={self.score:.3f})"


class Pattern:
    """Represents a query pattern for matching."""
    
    def __init__(self, atom_type: str, name: Optional[str] = None, 
                 outgoing: Optional[List['Pattern']] = None):
        self.atom_type = atom_type
        self.name = name
        self.outgoing = outgoing or []
        self.is_variable = name and name.startswith("$")
        self.variable_name = name[1:] if self.is_variable else None
    
    def __str__(self):
        if self.outgoing:
            outgoing_str = " ".join(str(p) for p in self.outgoing)
            return f"({self.atom_type} {outgoing_str})"
        else:
            return f"({self.atom_type} \"{self.name}\")" if self.name else f"({self.atom_type})"


class Query:
    """Represents a query with pattern and constraints."""
    
    def __init__(self, pattern: Pattern, 
                 constraints: Optional[List[Callable[[Dict[str, Atom]], bool]]] = None,
                 limit: int = 100):
        self.pattern = pattern
        self.constraints = constraints or []
        self.limit = limit
        self.variables: Set[str] = self._extract_variables(pattern)
    
    def _extract_variables(self, pattern: Pattern) -> Set[str]:
        """Extract all variables from the pattern."""
        variables = set()
        if pattern.is_variable:
            variables.add(pattern.variable_name)
        for outgoing in pattern.outgoing:
            variables.update(self._extract_variables(outgoing))
        return variables
    
    def add_constraint(self, constraint: Callable[[Dict[str, Atom]], bool]):
        """Add a constraint function to the query."""
        self.constraints.append(constraint)
    
    def __str__(self):
        return f"Query({self.pattern})"


class PatternMatcher:
    """Pattern matching engine for the AtomSpace."""
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
    
    def execute_query(self, query: Query) -> List[QueryResult]:
        """Execute a query and return matching results."""
        results = []
        self.atomspace.statistics['queries_executed'] += 1
        
        # Start matching from the root pattern
        bindings = {}
        self._match_pattern(query.pattern, None, bindings, results, query)
        
        # Apply constraints and limit results
        filtered_results = []
        for result in results:
            if self._check_constraints(result.bindings, query.constraints):
                filtered_results.append(result)
                if len(filtered_results) >= query.limit:
                    break
        
        return filtered_results
    
    def _match_pattern(self, pattern: Pattern, candidate: Optional[Atom], 
                      bindings: Dict[str, Atom], results: List[QueryResult], 
                      query: Query):
        """Recursively match a pattern against atoms."""
        
        if candidate is None:
            # Get candidate atoms from AtomSpace
            if pattern.is_variable:
                # Variable can match any atom of the specified type
                candidates = self.atomspace.get_atoms_by_type(pattern.atom_type)
            elif pattern.name:
                # Match by name and type
                candidates = [atom for atom in self.atomspace.get_atoms_by_name(pattern.name)
                            if atom.atom_type == pattern.atom_type]
            else:
                # Match by type only
                candidates = self.atomspace.get_atoms_by_type(pattern.atom_type)
            
            for candidate_atom in candidates:
                self._match_pattern(pattern, candidate_atom, bindings.copy(), results, query)
            return
        
        # Check if this atom matches the pattern
        if not self._atom_matches_pattern(candidate, pattern):
            return
        
        # Handle variable binding
        if pattern.is_variable:
            existing_binding = bindings.get(pattern.variable_name)
            if existing_binding is not None and existing_binding != candidate:
                return  # Variable already bound to different atom
            bindings[pattern.variable_name] = candidate
        
        # Match outgoing atoms for links
        if pattern.outgoing and candidate.is_link():
            if len(pattern.outgoing) != candidate.get_arity():
                return  # Arity mismatch
            
            self._match_outgoing(pattern.outgoing, candidate.outgoing_set, 0, 
                               bindings, results, query)
        elif not pattern.outgoing and candidate.is_node():
            # Node matched completely
            if self._all_variables_bound(query.variables, bindings):
                score = self._calculate_match_score(bindings)
                results.append(QueryResult(bindings.copy(), score))
        elif not pattern.outgoing and not candidate.outgoing_set:
            # Empty link or node
            if self._all_variables_bound(query.variables, bindings):
                score = self._calculate_match_score(bindings)
                results.append(QueryResult(bindings.copy(), score))
    
    def _match_outgoing(self, patterns: List[Pattern], atoms: List[Atom], 
                       index: int, bindings: Dict[str, Atom], 
                       results: List[QueryResult], query: Query):
        """Match outgoing patterns against outgoing atoms."""
        if index >= len(patterns):
            if self._all_variables_bound(query.variables, bindings):
                score = self._calculate_match_score(bindings)
                results.append(QueryResult(bindings.copy(), score))
            return
        
        pattern = patterns[index]
        atom = atoms[index]
        
        self._match_pattern(pattern, atom, bindings, [], query)
        
        # Continue with next outgoing pattern
        temp_results = []
        self._match_pattern(pattern, atom, bindings, temp_results, query)
        
        for temp_result in temp_results:
            self._match_outgoing(patterns, atoms, index + 1, temp_result.bindings,
                               results, query)
    
    def _atom_matches_pattern(self, atom: Atom, pattern: Pattern) -> bool:
        """Check if an atom matches a pattern."""
        if pattern.atom_type != atom.atom_type:
            return False
        
        if pattern.name and not pattern.is_variable and pattern.name != atom.name:
            return False
        
        return True
    
    def _all_variables_bound(self, variables: Set[str], bindings: Dict[str, Atom]) -> bool:
        """Check if all variables are bound."""
        return all(var in bindings for var in variables)
    
    def _calculate_match_score(self, bindings: Dict[str, Atom]) -> float:
        """Calculate a score for the match quality."""
        if not bindings:
            return 1.0
        
        total_confidence = sum(atom.truth_value.confidence for atom in bindings.values())
        total_strength = sum(atom.truth_value.strength for atom in bindings.values())
        
        avg_confidence = total_confidence / len(bindings)
        avg_strength = total_strength / len(bindings)
        
        return (avg_confidence + avg_strength) / 2.0
    
    def _check_constraints(self, bindings: Dict[str, Atom], 
                          constraints: List[Callable[[Dict[str, Atom]], bool]]) -> bool:
        """Check if bindings satisfy all constraints."""
        return all(constraint(bindings) for constraint in constraints)
    
    def find_similar_atoms(self, atom: Atom, similarity_threshold: float = 0.7) -> List[Atom]:
        """Find atoms similar to the given atom."""
        similar_atoms = []
        
        for candidate in self.atomspace.get_atoms_by_type(atom.atom_type):
            if candidate == atom:
                continue
            
            similarity = self._calculate_similarity(atom, candidate)
            if similarity >= similarity_threshold:
                similar_atoms.append(candidate)
        
        return sorted(similar_atoms, 
                     key=lambda a: self._calculate_similarity(atom, a), 
                     reverse=True)
    
    def _calculate_similarity(self, atom1: Atom, atom2: Atom) -> float:
        """Calculate similarity between two atoms."""
        if atom1.atom_type != atom2.atom_type:
            return 0.0
        
        # Name similarity (if both have names)
        name_sim = 0.0
        if atom1.name and atom2.name:
            name_sim = self._string_similarity(atom1.name, atom2.name)
        elif not atom1.name and not atom2.name:
            name_sim = 1.0
        
        # Structure similarity for links
        structure_sim = 0.0
        if atom1.is_link() and atom2.is_link():
            if atom1.get_arity() == atom2.get_arity():
                if atom1.get_arity() == 0:
                    structure_sim = 1.0
                else:
                    # Compare outgoing atoms
                    matches = 0
                    for a1 in atom1.outgoing_set:
                        for a2 in atom2.outgoing_set:
                            if a1.atom_type == a2.atom_type and a1.name == a2.name:
                                matches += 1
                                break
                    structure_sim = matches / atom1.get_arity()
        elif atom1.is_node() and atom2.is_node():
            structure_sim = 1.0
        
        # Truth value similarity
        tv_sim = 1.0 - abs(atom1.truth_value.strength - atom2.truth_value.strength)
        tv_sim *= 1.0 - abs(atom1.truth_value.confidence - atom2.truth_value.confidence)
        
        return (name_sim + structure_sim + tv_sim) / 3.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings."""
        if s1 == s2:
            return 1.0
        
        # Simple Levenshtein-like similarity
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        
        # Count common characters
        common_chars = 0
        for char in set(s1):
            common_chars += min(s1.count(char), s2.count(char))
        
        return common_chars / max_len
    
    def get_related_atoms(self, atom: Atom, max_depth: int = 2) -> Set[Atom]:
        """Get atoms related to the given atom through links."""
        related = set()
        to_visit = [(atom, 0)]
        visited = set()
        
        while to_visit:
            current_atom, depth = to_visit.pop(0)
            
            if current_atom in visited or depth > max_depth:
                continue
            
            visited.add(current_atom)
            related.add(current_atom)
            
            # Add atoms from incoming and outgoing sets
            for related_atom in current_atom.incoming_set:
                if related_atom not in visited:
                    to_visit.append((related_atom, depth + 1))
            
            for related_atom in current_atom.outgoing_set:
                if related_atom not in visited:
                    to_visit.append((related_atom, depth + 1))
        
        related.discard(atom)  # Remove the original atom
        return related