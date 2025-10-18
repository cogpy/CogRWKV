"""
Cognitive Architecture - High-level cognitive system integration
Orchestrates different cognitive subsystems for autonomous intelligent behavior.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from .atomspace import AtomSpace, Atom, Node, Link, AtomType, TruthValue
from .pattern_matcher import PatternMatcher, Query, Pattern
from .agent import AutonomousAgent, Goal, Action, GoalStatus


class CognitiveMode(Enum):
    """Different cognitive modes of operation."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    REFLECTION = "reflection"
    PROBLEM_SOLVING = "problem_solving"
    LEARNING = "learning"
    IDLE = "idle"


@dataclass
class CognitiveState:
    """Represents the current cognitive state."""
    mode: CognitiveMode = CognitiveMode.IDLE
    arousal: float = 0.5  # 0.0 to 1.0
    valence: float = 0.5  # 0.0 (negative) to 1.0 (positive)
    cognitive_load: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.5  # 0.0 to 1.0
    curiosity: float = 0.5  # 0.0 to 1.0
    
    def __str__(self):
        return f"CognitiveState({self.mode.value}, arousal={self.arousal:.2f}, valence={self.valence:.2f})"


class CognitiveProcess:
    """Base class for cognitive processes."""
    
    def __init__(self, name: str, priority: float = 0.5):
        self.name = name
        self.priority = priority
        self.is_active = False
        self.last_execution = 0
        self.execution_count = 0
    
    def can_execute(self, architecture: 'CognitiveArchitecture') -> bool:
        """Check if this process can execute in current context."""
        return True
    
    def execute(self, architecture: 'CognitiveArchitecture') -> Dict[str, Any]:
        """Execute the cognitive process."""
        self.is_active = True
        self.last_execution = time.time()
        self.execution_count += 1
        
        try:
            result = self._do_execute(architecture)
            return result
        finally:
            self.is_active = False
    
    def _do_execute(self, architecture: 'CognitiveArchitecture') -> Dict[str, Any]:
        """Actual execution logic - to be implemented by subclasses."""
        return {'status': 'completed'}


class PerceptionProcess(CognitiveProcess):
    """Process for perception and input processing."""
    
    def __init__(self):
        super().__init__("perception", priority=0.9)
    
    def can_execute(self, architecture: 'CognitiveArchitecture') -> bool:
        return len(architecture.input_buffer) > 0
    
    def _do_execute(self, architecture: 'CognitiveArchitecture') -> Dict[str, Any]:
        """Process inputs from the buffer."""
        processed = 0
        
        while architecture.input_buffer and processed < 5:  # Limit processing per cycle
            input_data = architecture.input_buffer.pop(0)
            
            # Convert input to atoms and add to AtomSpace
            if isinstance(input_data, str):
                # Text input - create concept nodes
                words = input_data.split()
                for word in words[:10]:  # Limit words processed
                    concept = architecture.atomspace.create_node(
                        AtomType.CONCEPT_NODE, word.lower(), 
                        TruthValue(0.8, 0.7)
                    )
                    architecture.agent.attention.focus_on(concept, 0.6)
                    
            elif isinstance(input_data, dict):
                # Structured input
                for key, value in input_data.items():
                    predicate = architecture.atomspace.create_node(
                        AtomType.PREDICATE_NODE, key, TruthValue(0.9, 0.8)
                    )
                    concept = architecture.atomspace.create_node(
                        AtomType.CONCEPT_NODE, str(value), TruthValue(0.8, 0.7)
                    )
                    evaluation = architecture.atomspace.create_link(
                        AtomType.EVALUATION_LINK, [predicate, concept],
                        TruthValue(0.8, 0.8)
                    )
                    architecture.agent.attention.focus_on(evaluation, 0.7)
            
            processed += 1
        
        return {'inputs_processed': processed}


class ReasoningProcess(CognitiveProcess):
    """Process for reasoning and inference."""
    
    def __init__(self):
        super().__init__("reasoning", priority=0.7)
    
    def can_execute(self, architecture: 'CognitiveArchitecture') -> bool:
        return len(architecture.agent.attention.focus_atoms) > 0
    
    def _do_execute(self, architecture: 'CognitiveArchitecture') -> Dict[str, Any]:
        """Perform reasoning on focused atoms."""
        inferences_made = 0
        focused_atoms = architecture.agent.attention.get_focused_atoms()
        
        # Simple inheritance reasoning
        for atom in focused_atoms[:3]:  # Limit reasoning scope
            if atom.atom_type == AtomType.CONCEPT_NODE:
                # Look for inheritance relationships
                pattern = Pattern(
                    AtomType.INHERITANCE_LINK,
                    outgoing=[
                        Pattern(AtomType.CONCEPT_NODE, atom.name),
                        Pattern(AtomType.CONCEPT_NODE, "$parent")
                    ]
                )
                
                query = Query(pattern, limit=3)
                results = architecture.pattern_matcher.execute_query(query)
                
                for result in results:
                    parent = result.bindings.get('parent')
                    if parent:
                        architecture.agent.attention.focus_on(parent, 0.5)
                        inferences_made += 1
        
        # Similarity reasoning
        for atom in focused_atoms[:2]:
            similar_atoms = architecture.pattern_matcher.find_similar_atoms(
                atom, similarity_threshold=0.6
            )
            for similar in similar_atoms[:2]:
                architecture.agent.attention.focus_on(similar, 0.3)
                inferences_made += 1
        
        return {'inferences_made': inferences_made}


class PlanningProcess(CognitiveProcess):
    """Process for planning and goal management."""
    
    def __init__(self):
        super().__init__("planning", priority=0.6)
    
    def can_execute(self, architecture: 'CognitiveArchitecture') -> bool:
        active_goals = [g for g in architecture.agent.goals.values() 
                       if g.status == GoalStatus.ACTIVE]
        return len(active_goals) > 0 or len(architecture.agent.action_queue) < 2
    
    def _do_execute(self, architecture: 'CognitiveArchitecture') -> Dict[str, Any]:
        """Plan actions for goals."""
        plans_created = 0
        
        # Find goals that need planning
        for goal in architecture.agent.goals.values():
            if goal.status == GoalStatus.PENDING:
                goal.status = GoalStatus.ACTIVE
                
                # Create simple plans based on goal description
                if "learn" in goal.description.lower():
                    # Learning goal - schedule learning actions
                    learn_action = next((a for a in architecture.agent.actions.values() 
                                       if a.name == "learn"), None)
                    if learn_action:
                        architecture.agent.schedule_action(learn_action.id, goal.id)
                        plans_created += 1
                
                elif "explore" in goal.description.lower():
                    # Exploration goal - schedule thinking actions
                    think_action = next((a for a in architecture.agent.actions.values() 
                                       if a.name == "think"), None)
                    if think_action:
                        architecture.agent.schedule_action(think_action.id, goal.id)
                        plans_created += 1
        
        return {'plans_created': plans_created}


class LearningProcess(CognitiveProcess):
    """Process for learning and knowledge update."""
    
    def __init__(self):
        super().__init__("learning", priority=0.5)
    
    def can_execute(self, architecture: 'CognitiveArchitecture') -> bool:
        return architecture.state.cognitive_load < 0.8  # Don't learn when overloaded
    
    def _do_execute(self, architecture: 'CognitiveArchitecture') -> Dict[str, Any]:
        """Perform learning operations."""
        updates_made = 0
        
        # Strengthen frequently accessed atoms
        for atom in architecture.agent.attention.get_focused_atoms():
            if atom.attention_value > 0.5:
                tv = atom.truth_value
                new_confidence = min(1.0, tv.confidence + 0.02)
                atom.truth_value = TruthValue(tv.strength, new_confidence)
                updates_made += 1
        
        # Create associations between co-occurring atoms
        focused = architecture.agent.attention.get_focused_atoms()
        if len(focused) >= 2:
            for i in range(min(3, len(focused))):
                for j in range(i + 1, min(i + 3, len(focused))):
                    atom1, atom2 = focused[i], focused[j]
                    
                    # Create similarity link if not exists
                    existing_similarity = False
                    for incoming in atom1.incoming_set:
                        if (incoming.atom_type == AtomType.SIMILARITY_LINK and
                            atom2 in incoming.outgoing_set):
                            existing_similarity = True
                            break
                    
                    if not existing_similarity:
                        similarity_link = architecture.atomspace.create_link(
                            AtomType.SIMILARITY_LINK, [atom1, atom2],
                            TruthValue(0.4, 0.6)
                        )
                        updates_made += 1
        
        return {'learning_updates': updates_made}


class MetacognitionProcess(CognitiveProcess):
    """Process for metacognitive monitoring and control."""
    
    def __init__(self):
        super().__init__("metacognition", priority=0.4)
        self.last_state_update = 0
    
    def can_execute(self, architecture: 'CognitiveArchitecture') -> bool:
        return time.time() - self.last_state_update > 2.0  # Update every 2 seconds
    
    def _do_execute(self, architecture: 'CognitiveArchitecture') -> Dict[str, Any]:
        """Monitor and adjust cognitive state."""
        self.last_state_update = time.time()
        
        # Update cognitive load based on activity
        active_processes = sum(1 for p in architecture.processes.values() if p.is_active)
        focused_atoms = len(architecture.agent.attention.focus_atoms)
        pending_actions = len(architecture.agent.action_queue)
        
        cognitive_load = (active_processes * 0.2 + focused_atoms * 0.01 + pending_actions * 0.1)
        architecture.state.cognitive_load = min(1.0, cognitive_load)
        
        # Update arousal based on goal activity
        active_goals = sum(1 for g in architecture.agent.goals.values() 
                          if g.status == GoalStatus.ACTIVE)
        architecture.state.arousal = min(1.0, 0.3 + active_goals * 0.2)
        
        # Update confidence based on recent successes
        recent_episodes = architecture.agent.memory.recall_episodes(
            {'type': 'goal_completed'}, limit=5
        )
        if len(recent_episodes) > 0:
            architecture.state.confidence = min(1.0, architecture.state.confidence + 0.1)
        else:
            architecture.state.confidence = max(0.0, architecture.state.confidence - 0.05)
        
        # Determine cognitive mode
        if architecture.state.cognitive_load > 0.8:
            architecture.state.mode = CognitiveMode.REFLECTION
        elif architecture.state.arousal > 0.7:
            architecture.state.mode = CognitiveMode.PROBLEM_SOLVING
        elif architecture.state.curiosity > 0.6:
            architecture.state.mode = CognitiveMode.EXPLORATION
        elif focused_atoms > 10:
            architecture.state.mode = CognitiveMode.LEARNING
        else:
            architecture.state.mode = CognitiveMode.IDLE
        
        return {
            'cognitive_load': architecture.state.cognitive_load,
            'arousal': architecture.state.arousal,
            'mode': architecture.state.mode.value
        }


class CognitiveArchitecture:
    """
    High-level cognitive architecture that orchestrates different cognitive processes.
    """
    
    def __init__(self, agent_name: str = "CogAgent"):
        self.atomspace = AtomSpace()
        self.agent = AutonomousAgent(agent_name, self.atomspace)
        self.pattern_matcher = PatternMatcher(self.atomspace)
        
        # Cognitive state and processes
        self.state = CognitiveState()
        self.processes: Dict[str, CognitiveProcess] = {
            'perception': PerceptionProcess(),
            'reasoning': ReasoningProcess(),
            'planning': PlanningProcess(),
            'learning': LearningProcess(),
            'metacognition': MetacognitionProcess()
        }
        
        # Input/output buffers
        self.input_buffer: List[Any] = []
        self.output_buffer: List[Any] = []
        
        # Control parameters
        self.cycle_interval = 1.0  # seconds
        self.max_cycles_per_second = 10
        
        # Statistics
        self.statistics = {
            'cognitive_cycles': 0,
            'total_processes_executed': 0,
            'last_cycle_duration': 0.0,
            'average_cycle_duration': 0.0
        }
        
        # Threading
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
    
    def add_input(self, input_data: Any):
        """Add input to the processing buffer."""
        self.input_buffer.append(input_data)
    
    def get_output(self) -> List[Any]:
        """Get and clear the output buffer."""
        output = self.output_buffer.copy()
        self.output_buffer.clear()
        return output
    
    def add_goal(self, description: str, priority: float = 0.5, 
                deadline: Optional[float] = None) -> str:
        """Add a goal to the system."""
        goal = Goal(
            description=description,
            priority=priority,
            deadline=deadline
        )
        self.agent.add_goal(goal)
        return goal.id
    
    def add_knowledge(self, knowledge: Any):
        """Add knowledge to the system."""
        self.agent.add_knowledge(knowledge)
    
    def cognitive_cycle(self) -> Dict[str, Any]:
        """Execute one cognitive cycle."""
        cycle_start = time.time()
        self.statistics['cognitive_cycles'] += 1
        
        cycle_results = {}
        processes_executed = 0
        
        # Execute processes in priority order
        sorted_processes = sorted(self.processes.values(), 
                                key=lambda p: p.priority, reverse=True)
        
        for process in sorted_processes:
            if process.can_execute(self):
                result = process.execute(self)
                cycle_results[process.name] = result
                processes_executed += 1
                self.statistics['total_processes_executed'] += 1
                
                # Limit processes per cycle to prevent overload
                if processes_executed >= 3:
                    break
        
        # Execute agent's cognitive cycle
        self.agent.cognitive_cycle()
        
        # Update timing statistics
        cycle_duration = time.time() - cycle_start
        self.statistics['last_cycle_duration'] = cycle_duration
        
        if self.statistics['cognitive_cycles'] > 0:
            total_duration = (self.statistics['average_cycle_duration'] * 
                            (self.statistics['cognitive_cycles'] - 1) + cycle_duration)
            self.statistics['average_cycle_duration'] = total_duration / self.statistics['cognitive_cycles']
        
        cycle_results['cycle_info'] = {
            'processes_executed': processes_executed,
            'duration': cycle_duration,
            'state': str(self.state)
        }
        
        return cycle_results
    
    def start(self):
        """Start the cognitive architecture."""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start the agent
        self.agent.start()
        
        # Start cognitive processing thread
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        print(f"Cognitive architecture started for agent '{self.agent.name}'")
    
    def stop(self):
        """Stop the cognitive architecture."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Stop the agent
        self.agent.stop()
        
        # Stop cognitive processing thread
        if self.thread:
            self.thread.join(timeout=5.0)
        
        print(f"Cognitive architecture stopped for agent '{self.agent.name}'")
    
    def _run_loop(self):
        """Main cognitive processing loop."""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Execute cognitive cycle
                cycle_results = self.cognitive_cycle()
                
                # Add cycle results to output buffer for monitoring
                if any(cycle_results.values()):  # Only add if something happened
                    self.output_buffer.append({
                        'type': 'cognitive_cycle',
                        'timestamp': time.time(),
                        'results': cycle_results
                    })
                
                # Wait for next cycle
                self.stop_event.wait(self.cycle_interval)
                
            except Exception as e:
                print(f"Error in cognitive cycle: {e}")
                self.stop_event.wait(1.0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'is_running': self.is_running,
            'state': {
                'mode': self.state.mode.value,
                'arousal': self.state.arousal,
                'cognitive_load': self.state.cognitive_load,
                'confidence': self.state.confidence
            },
            'agent_status': self.agent.get_status(),
            'statistics': self.statistics.copy(),
            'processes': {name: {
                'active': proc.is_active,
                'executions': proc.execution_count,
                'last_execution': proc.last_execution
            } for name, proc in self.processes.items()},
            'buffers': {
                'input_size': len(self.input_buffer),
                'output_size': len(self.output_buffer)
            }
        }
    
    def __str__(self):
        return f"CognitiveArchitecture('{self.agent.name}', {self.state.mode.value})"