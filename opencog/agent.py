"""
Autonomous Agent Implementation
Goal-directed autonomous agents with cognitive capabilities.
"""

import time
import threading
from typing import List, Dict, Set, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from .atomspace import AtomSpace, Atom, Node, Link, AtomType, TruthValue
from .pattern_matcher import PatternMatcher, Query, Pattern


class GoalStatus(Enum):
    """Status of a goal."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionStatus(Enum):
    """Status of an action."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Goal:
    """Represents a goal for the autonomous agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: float = 0.5  # 0.0 to 1.0
    status: GoalStatus = GoalStatus.PENDING
    created_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    success_conditions: List[Callable[[], bool]] = field(default_factory=list)
    failure_conditions: List[Callable[[], bool]] = field(default_factory=list)
    parent_goal_id: Optional[str] = None
    sub_goal_ids: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if goal has expired."""
        return self.deadline is not None and time.time() > self.deadline
    
    def check_success(self) -> bool:
        """Check if goal is successfully completed."""
        if not self.success_conditions:
            return False
        return all(condition() for condition in self.success_conditions)
    
    def check_failure(self) -> bool:
        """Check if goal has failed."""
        return any(condition() for condition in self.failure_conditions)
    
    def __str__(self):
        return f"Goal({self.description}, {self.status.value}, priority={self.priority})"


@dataclass 
class Action:
    """Represents an action that can be executed by the agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    executor: Optional[Callable[['AutonomousAgent'], Any]] = None
    status: ActionStatus = ActionStatus.PENDING
    goal_id: Optional[str] = None
    prerequisites: List[str] = field(default_factory=list)  # Action IDs
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def execute(self, agent: 'AutonomousAgent') -> bool:
        """Execute the action."""
        if self.executor is None:
            self.status = ActionStatus.FAILED
            self.error = "No executor defined"
            return False
        
        try:
            self.status = ActionStatus.EXECUTING
            self.start_time = time.time()
            self.result = self.executor(agent)
            self.status = ActionStatus.COMPLETED
            self.end_time = time.time()
            return True
        except Exception as e:
            self.status = ActionStatus.FAILED
            self.error = str(e)
            self.end_time = time.time()
            return False
    
    def __str__(self):
        return f"Action({self.name}, {self.status.value})"


class Memory:
    """Agent's memory system."""
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.episodic_memory: List[Dict[str, Any]] = []
        self.working_memory: Dict[str, Any] = {}
        self.semantic_memory = atomspace  # Use AtomSpace for semantic memory
        
    def add_episode(self, event: Dict[str, Any]):
        """Add an episodic memory."""
        episode = {
            'timestamp': time.time(),
            'event': event,
            'id': str(uuid.uuid4())
        }
        self.episodic_memory.append(episode)
        
        # Limit memory size
        if len(self.episodic_memory) > 1000:
            self.episodic_memory = self.episodic_memory[-1000:]
    
    def recall_episodes(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Recall episodic memories matching query."""
        matches = []
        for episode in reversed(self.episodic_memory):
            if self._matches_query(episode['event'], query):
                matches.append(episode)
                if len(matches) >= limit:
                    break
        return matches
    
    def _matches_query(self, event: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if event matches query criteria."""
        for key, value in query.items():
            if key not in event or event[key] != value:
                return False
        return True
    
    def store_knowledge(self, knowledge: Atom):
        """Store knowledge in semantic memory."""
        self.atomspace.add_atom(knowledge)
    
    def retrieve_knowledge(self, pattern: Pattern) -> List[Atom]:
        """Retrieve knowledge from semantic memory."""
        matcher = PatternMatcher(self.atomspace)
        query = Query(pattern)
        results = matcher.execute_query(query)
        return [result.bindings for result in results]


class AttentionSystem:
    """Attention allocation system."""
    
    def __init__(self):
        self.focus_atoms: Set[Atom] = set()
        self.attention_values: Dict[str, float] = {}
        self.max_focus_size = 20
    
    def focus_on(self, atom: Atom, strength: float = 1.0):
        """Focus attention on an atom."""
        self.focus_atoms.add(atom)
        self.attention_values[atom.id] = strength
        atom.attention_value = strength
        
        # Limit focus size
        if len(self.focus_atoms) > self.max_focus_size:
            # Remove least attended atom
            min_atom = min(self.focus_atoms, key=lambda a: a.attention_value)
            self.unfocus(min_atom)
    
    def unfocus(self, atom: Atom):
        """Remove attention from an atom."""
        self.focus_atoms.discard(atom)
        self.attention_values.pop(atom.id, None)
        atom.attention_value = 0.0
    
    def get_focused_atoms(self) -> List[Atom]:
        """Get currently focused atoms, sorted by attention value."""
        return sorted(self.focus_atoms, key=lambda a: a.attention_value, reverse=True)
    
    def update_attention(self):
        """Update attention values based on relevance."""
        # Decay attention values
        decay_rate = 0.95
        for atom in list(self.focus_atoms):
            new_value = self.attention_values.get(atom.id, 0) * decay_rate
            if new_value < 0.1:
                self.unfocus(atom)
            else:
                self.attention_values[atom.id] = new_value
                atom.attention_value = new_value


class AutonomousAgent:
    """Autonomous agent with cognitive capabilities."""
    
    def __init__(self, name: str, atomspace: Optional[AtomSpace] = None):
        self.name = name
        self.atomspace = atomspace or AtomSpace()
        self.pattern_matcher = PatternMatcher(self.atomspace)
        self.memory = Memory(self.atomspace)
        self.attention = AttentionSystem()
        
        # Goals and actions
        self.goals: Dict[str, Goal] = {}
        self.actions: Dict[str, Action] = {}
        self.action_queue: List[str] = []
        
        # Agent state
        self.is_running = False
        self.cognitive_cycle_interval = 1.0  # seconds
        self.last_cycle_time = 0
        
        # Statistics
        self.statistics = {
            'cycles_completed': 0,
            'goals_completed': 0,
            'actions_executed': 0,
            'knowledge_items': 0
        }
        
        # Thread management
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Built-in actions
        self._register_built_in_actions()
    
    def _register_built_in_actions(self):
        """Register built-in actions."""
        
        def think_action(agent):
            """Basic thinking/reasoning action."""
            # Get focused atoms and perform basic reasoning
            focused = agent.attention.get_focused_atoms()
            if focused:
                # Simple association and pattern matching
                for atom in focused[:5]:  # Limit processing
                    related = agent.pattern_matcher.get_related_atoms(atom, max_depth=1)
                    for related_atom in list(related)[:3]:  # Limit associations
                        agent.attention.focus_on(related_atom, 0.3)
            
            return f"Performed thinking cycle with {len(focused)} focused concepts"
        
        def learn_action(agent):
            """Learning action - strengthen relevant connections."""
            focused = agent.attention.get_focused_atoms()
            for atom in focused:
                # Strengthen truth values of focused atoms
                tv = atom.truth_value
                new_strength = min(1.0, tv.strength + 0.1)
                new_confidence = min(1.0, tv.confidence + 0.05)
                atom.truth_value = TruthValue(new_strength, new_confidence)
            
            return f"Learning reinforced {len(focused)} concepts"
        
        self.register_action("think", think_action, "Perform basic reasoning and association")
        self.register_action("learn", learn_action, "Strengthen knowledge connections")
    
    def add_goal(self, goal: Goal):
        """Add a goal to the agent."""
        self.goals[goal.id] = goal
        self.memory.add_episode({
            'type': 'goal_added',
            'goal_id': goal.id,
            'description': goal.description
        })
    
    def remove_goal(self, goal_id: str):
        """Remove a goal from the agent."""
        if goal_id in self.goals:
            goal = self.goals.pop(goal_id)
            self.memory.add_episode({
                'type': 'goal_removed',
                'goal_id': goal_id,
                'final_status': goal.status.value
            })
    
    def register_action(self, name: str, executor: Callable[['AutonomousAgent'], Any], 
                       description: str = "", action_id: Optional[str] = None):
        """Register an action that the agent can perform."""
        action = Action(
            id=action_id or str(uuid.uuid4()),
            name=name,
            description=description,
            executor=executor
        )
        self.actions[action.id] = action
        return action.id
    
    def schedule_action(self, action_id: str, goal_id: Optional[str] = None):
        """Schedule an action for execution."""
        if action_id in self.actions:
            action = self.actions[action_id]
            action.goal_id = goal_id
            self.action_queue.append(action_id)
    
    def execute_next_action(self) -> bool:
        """Execute the next action in the queue."""
        if not self.action_queue:
            return False
        
        action_id = self.action_queue.pop(0)
        action = self.actions.get(action_id)
        
        if action is None:
            return False
        
        success = action.execute(self)
        self.statistics['actions_executed'] += 1
        
        self.memory.add_episode({
            'type': 'action_executed',
            'action_id': action_id,
            'action_name': action.name,
            'success': success,
            'result': action.result
        })
        
        return success
    
    def cognitive_cycle(self):
        """Perform one cognitive cycle."""
        self.statistics['cycles_completed'] += 1
        self.last_cycle_time = time.time()
        
        # Update attention system
        self.attention.update_attention()
        
        # Evaluate goals
        self._evaluate_goals()
        
        # Plan and execute actions
        self._plan_actions()
        
        # Execute one action if available
        if self.action_queue:
            self.execute_next_action()
        
        # Update statistics
        self.statistics['knowledge_items'] = self.atomspace.size()
    
    def _evaluate_goals(self):
        """Evaluate current goals and update their status."""
        for goal in list(self.goals.values()):
            if goal.status == GoalStatus.ACTIVE or goal.status == GoalStatus.PENDING:
                if goal.is_expired():
                    goal.status = GoalStatus.FAILED
                    self.memory.add_episode({
                        'type': 'goal_expired',
                        'goal_id': goal.id
                    })
                elif goal.check_success():
                    goal.status = GoalStatus.COMPLETED
                    self.statistics['goals_completed'] += 1
                    self.memory.add_episode({
                        'type': 'goal_completed',
                        'goal_id': goal.id
                    })
                elif goal.check_failure():
                    goal.status = GoalStatus.FAILED
                    self.memory.add_episode({
                        'type': 'goal_failed',
                        'goal_id': goal.id
                    })
    
    def _plan_actions(self):
        """Plan actions based on current goals."""
        # Simple planning: schedule basic actions for active goals
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        
        if not active_goals and len(self.action_queue) < 2:
            # No active goals, perform basic cognitive actions
            think_action = next((a for a in self.actions.values() if a.name == "think"), None)
            learn_action = next((a for a in self.actions.values() if a.name == "learn"), None)
            
            if think_action and think_action.id not in self.action_queue:
                self.schedule_action(think_action.id)
            if learn_action and learn_action.id not in self.action_queue:
                self.schedule_action(learn_action.id)
    
    def start(self):
        """Start the autonomous agent."""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        print(f"Autonomous agent '{self.name}' started")
    
    def stop(self):
        """Stop the autonomous agent."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.thread:
            self.thread.join(timeout=5.0)
        
        print(f"Autonomous agent '{self.name}' stopped")
    
    def _run_loop(self):
        """Main execution loop."""
        while self.is_running and not self.stop_event.is_set():
            try:
                self.cognitive_cycle()
                self.stop_event.wait(self.cognitive_cycle_interval)
            except Exception as e:
                print(f"Error in cognitive cycle: {e}")
                self.stop_event.wait(1.0)
    
    def add_knowledge(self, knowledge: Union[Atom, str]):
        """Add knowledge to the agent."""
        if isinstance(knowledge, str):
            # Create a concept node for string knowledge
            concept = self.atomspace.create_node(AtomType.CONCEPT_NODE, knowledge)
            self.attention.focus_on(concept, 0.8)
        else:
            self.atomspace.add_atom(knowledge)
            self.attention.focus_on(knowledge, 0.8)
        
        self.memory.add_episode({
            'type': 'knowledge_added',
            'knowledge': str(knowledge)
        })
    
    def query_knowledge(self, pattern: Pattern) -> List[Dict[str, Any]]:
        """Query the agent's knowledge."""
        query = Query(pattern)
        results = self.pattern_matcher.execute_query(query)
        return [{'bindings': r.bindings, 'score': r.score} for r in results]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'name': self.name,
            'is_running': self.is_running,
            'statistics': self.statistics.copy(),
            'goals': {gid: g.status.value for gid, g in self.goals.items()},
            'actions_queued': len(self.action_queue),
            'focused_atoms': len(self.attention.focus_atoms),
            'knowledge_size': self.atomspace.size()
        }
    
    def __str__(self):
        return f"AutonomousAgent('{self.name}', {len(self.goals)} goals, {self.atomspace.size()} knowledge items)"