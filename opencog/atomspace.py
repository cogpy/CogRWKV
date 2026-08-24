"""
AtomSpace Implementation - Knowledge Representation System
Inspired by OpenCog's AtomSpace for hypergraph knowledge representation.
"""

import uuid
import threading
from typing import Dict, List, Set, Optional, Union, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import json


@dataclass
class TruthValue:
    """Represents truth value with strength and confidence."""
    strength: float = 0.5  # 0.0 to 1.0
    confidence: float = 0.5  # 0.0 to 1.0
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def __str__(self):
        return f"TV({self.strength:.3f}, {self.confidence:.3f})"


class AtomType:
    """Defines different types of atoms."""
    ATOM = "Atom"
    NODE = "Node"
    LINK = "Link"
    CONCEPT_NODE = "ConceptNode"
    PREDICATE_NODE = "PredicateNode"
    VARIABLE_NODE = "VariableNode"
    EVALUATION_LINK = "EvaluationLink"
    INHERITANCE_LINK = "InheritanceLink"
    SIMILARITY_LINK = "SimilarityLink"
    IMPLICATION_LINK = "ImplicationLink"
    AND_LINK = "AndLink"
    OR_LINK = "OrLink"
    NOT_LINK = "NotLink"


class Atom:
    """Base class for all atoms in the AtomSpace."""
    
    def __init__(self, atom_type: str, name: Optional[str] = None, 
                 truth_value: Optional[TruthValue] = None):
        self.id = str(uuid.uuid4())
        self.atom_type = atom_type
        self.name = name or ""
        self.truth_value = truth_value or TruthValue()
        self.incoming_set: Set['Atom'] = set()
        self.outgoing_set: List['Atom'] = []
        self.attention_value = 0.5
        self.metadata: Dict[str, Any] = {}
    
    def add_incoming(self, atom: 'Atom'):
        """Add atom to incoming set."""
        self.incoming_set.add(atom)
    
    def remove_incoming(self, atom: 'Atom'):
        """Remove atom from incoming set."""
        self.incoming_set.discard(atom)
    
    def add_outgoing(self, atom: 'Atom'):
        """Add atom to outgoing set."""
        self.outgoing_set.append(atom)
        atom.add_incoming(self)
    
    def get_arity(self) -> int:
        """Get number of outgoing atoms."""
        return len(self.outgoing_set)
    
    def is_node(self) -> bool:
        """Check if this is a Node."""
        return self.get_arity() == 0
    
    def is_link(self) -> bool:
        """Check if this is a Link."""
        return self.get_arity() > 0
    
    def __str__(self):
        if self.is_node():
            return f"({self.atom_type} \"{self.name}\" {self.truth_value})"
        else:
            outgoing_str = " ".join(str(atom) for atom in self.outgoing_set)
            return f"({self.atom_type} {outgoing_str} {self.truth_value})"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Atom) and self.id == other.id


class Node(Atom):
    """Node atom with a name."""
    
    def __init__(self, atom_type: str = AtomType.NODE, name: str = "", 
                 truth_value: Optional[TruthValue] = None):
        super().__init__(atom_type, name, truth_value)


class Link(Atom):
    """Link atom connecting other atoms."""
    
    def __init__(self, atom_type: str = AtomType.LINK, 
                 outgoing: Optional[List[Atom]] = None,
                 truth_value: Optional[TruthValue] = None):
        super().__init__(atom_type, "", truth_value)
        if outgoing:
            for atom in outgoing:
                self.add_outgoing(atom)


class AtomSpace:
    """
    AtomSpace - Hypergraph knowledge representation system.
    Stores and manages atoms (nodes and links) with pattern matching capabilities.
    """
    
    def __init__(self):
        self.atoms: Dict[str, Atom] = {}
        self.atoms_by_type: Dict[str, Set[Atom]] = defaultdict(set)
        self.atoms_by_name: Dict[str, Set[Atom]] = defaultdict(set)
        self.lock = threading.RLock()
        self.statistics = {
            'total_atoms': 0,
            'nodes': 0,
            'links': 0,
            'queries_executed': 0
        }
    
    def add_atom(self, atom: Atom) -> Atom:
        """Add an atom to the AtomSpace."""
        with self.lock:
            if atom.id in self.atoms:
                return self.atoms[atom.id]
            
            self.atoms[atom.id] = atom
            self.atoms_by_type[atom.atom_type].add(atom)
            if atom.name:
                self.atoms_by_name[atom.name].add(atom)
            
            self.statistics['total_atoms'] += 1
            if atom.is_node():
                self.statistics['nodes'] += 1
            else:
                self.statistics['links'] += 1
            
            return atom
    
    def remove_atom(self, atom: Atom) -> bool:
        """Remove an atom from the AtomSpace."""
        with self.lock:
            if atom.id not in self.atoms:
                return False
            
            # Remove from incoming sets
            for incoming in atom.incoming_set:
                if atom in incoming.outgoing_set:
                    incoming.outgoing_set.remove(atom)
            
            # Remove from outgoing sets
            for outgoing in atom.outgoing_set:
                outgoing.remove_incoming(atom)
            
            # Remove from indices
            del self.atoms[atom.id]
            self.atoms_by_type[atom.atom_type].discard(atom)
            if atom.name:
                self.atoms_by_name[atom.name].discard(atom)
            
            self.statistics['total_atoms'] -= 1
            if atom.is_node():
                self.statistics['nodes'] -= 1
            else:
                self.statistics['links'] -= 1
            
            return True
    
    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Get atom by ID."""
        return self.atoms.get(atom_id)
    
    def get_atoms_by_type(self, atom_type: str) -> Set[Atom]:
        """Get all atoms of a specific type."""
        return self.atoms_by_type.get(atom_type, set()).copy()
    
    def get_atoms_by_name(self, name: str) -> Set[Atom]:
        """Get all atoms with a specific name."""
        return self.atoms_by_name.get(name, set()).copy()
    
    def get_all_atoms(self) -> List[Atom]:
        """Get all atoms in the AtomSpace."""
        return list(self.atoms.values())
    
    def create_node(self, atom_type: str, name: str, 
                   truth_value: Optional[TruthValue] = None) -> Node:
        """Create and add a node to the AtomSpace. Returns existing node if duplicate found."""
        # Check for existing node with same type and name
        existing_nodes = self.get_atoms_by_name(name)
        for existing in existing_nodes:
            if existing.atom_type == atom_type:
                # Update truth value if provided and merge with existing
                if truth_value:
                    existing_tv = existing.truth_value
                    # Merge truth values by taking weighted average
                    combined_strength = (existing_tv.strength * existing_tv.confidence + 
                                       truth_value.strength * truth_value.confidence) / (
                                       existing_tv.confidence + truth_value.confidence + 1e-6)
                    combined_confidence = min(1.0, existing_tv.confidence + truth_value.confidence)
                    existing.truth_value = TruthValue(combined_strength, combined_confidence)
                return existing
        
        # Create new node if no duplicate found
        node = Node(atom_type, name, truth_value)
        return self.add_atom(node)
    
    def create_link(self, atom_type: str, outgoing: List[Atom],
                   truth_value: Optional[TruthValue] = None) -> Link:
        """Create and add a link to the AtomSpace."""
        link = Link(atom_type, outgoing, truth_value)
        return self.add_atom(link)
    
    def size(self) -> int:
        """Get total number of atoms."""
        return len(self.atoms)
    
    def clear(self):
        """Clear all atoms from the AtomSpace."""
        with self.lock:
            self.atoms.clear()
            self.atoms_by_type.clear()
            self.atoms_by_name.clear()
            self.statistics = {
                'total_atoms': 0,
                'nodes': 0,
                'links': 0,
                'queries_executed': 0
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get AtomSpace statistics."""
        return self.statistics.copy()
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export AtomSpace to dictionary."""
        atoms_data = []
        for atom in self.atoms.values():
            atom_data = {
                'id': atom.id,
                'type': atom.atom_type,
                'name': atom.name,
                'truth_value': {
                    'strength': atom.truth_value.strength,
                    'confidence': atom.truth_value.confidence
                },
                'outgoing': [out_atom.id for out_atom in atom.outgoing_set],
                'metadata': atom.metadata
            }
            atoms_data.append(atom_data)
        
        return {
            'atoms': atoms_data,
            'statistics': self.statistics
        }
    
    def import_from_dict(self, data: Dict[str, Any]):
        """Import AtomSpace from dictionary."""
        self.clear()
        
        # First pass: create all atoms without connections
        atom_map = {}
        for atom_data in data.get('atoms', []):
            if atom_data.get('outgoing'):
                # This is a link, create it later
                continue
            else:
                # This is a node
                truth_value = TruthValue(
                    atom_data['truth_value']['strength'],
                    atom_data['truth_value']['confidence']
                )
                node = Node(atom_data['type'], atom_data['name'], truth_value)
                node.id = atom_data['id']  # Preserve original ID
                node.metadata = atom_data.get('metadata', {})
                atom_map[node.id] = node
                self.add_atom(node)
        
        # Second pass: create links iteratively (handles link-to-link references)
        pending_links = [ad for ad in data.get('atoms', []) if ad.get('outgoing')]
        while pending_links:
            remaining = []
            for atom_data in pending_links:
                if all(aid in atom_map for aid in atom_data['outgoing']):
                    truth_value = TruthValue(
                        atom_data['truth_value']['strength'],
                        atom_data['truth_value']['confidence']
                    )
                    outgoing_atoms = [atom_map[atom_id] for atom_id in atom_data['outgoing']]
                    link = Link(atom_data['type'], outgoing_atoms, truth_value)
                    link.id = atom_data['id']  # Preserve original ID
                    link.metadata = atom_data.get('metadata', {})
                    atom_map[link.id] = link
                    self.add_atom(link)
                else:
                    remaining.append(atom_data)
            if len(remaining) == len(pending_links):
                break  # No progress — unresolvable references
            pending_links = remaining
    
    def __str__(self):
        return f"AtomSpace({self.size()} atoms)"
    
    def __repr__(self):
        return self.__str__()