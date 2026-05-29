"""
RWKV Integration - Bridges OpenCog cognitive architecture with RWKV language model
Provides seamless integration between symbolic reasoning and neural language processing.
"""

import os
import sys
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass
import json

# Add RWKV package to path
current_path = os.path.dirname(os.path.abspath(__file__))
rwkv_path = os.path.join(current_path, '..', 'rwkv_pip_package', 'src')
sys.path.append(rwkv_path)

try:
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE
    RWKV_AVAILABLE = True
except ImportError:
    # Fallback for when RWKV is not available
    RWKV_AVAILABLE = False
    print("RWKV model not available, using mock implementation")

from .atomspace import AtomSpace, Atom, Node, Link, AtomType, TruthValue
from .pattern_matcher import PatternMatcher, Query, Pattern
from .agent import AutonomousAgent, Goal, Action, ActionStatus
from .cognitive_architecture import CognitiveArchitecture, CognitiveProcess


@dataclass
class RWKVResponse:
    """Response from RWKV model."""
    text: str
    tokens: List[int]
    logits: Optional[List[float]] = None
    confidence: float = 0.5
    processing_time: float = 0.0


class MockRWKV:
    """Mock RWKV implementation for testing when model is not available."""
    
    def __init__(self, model_path: str, strategy: str = 'cpu fp32'):
        self.model_path = model_path
        self.strategy = strategy
        self.loaded = True
        
    def forward(self, tokens: List[int], state=None):
        # Return mock logits and state
        import torch
        mock_logits = torch.randn(50277)  # Standard RWKV vocab size
        mock_state = torch.randn(32, 512) if state is None else state
        return mock_logits, mock_state


class MockPipeline:
    """Mock pipeline for testing."""
    
    def __init__(self, model, tokenizer_path: str):
        self.model = model
        self.tokenizer_path = tokenizer_path
        
    def generate(self, prompt: str, token_count: int = 100, 
                temperature: float = 1.0, top_p: float = 0.8) -> str:
        # Return mock response
        return f"Mock response to: {prompt[:50]}..."
    
    def encode(self, text: str) -> List[int]:
        # Mock tokenization
        return [i for i in range(len(text.split()))]
    
    def decode(self, tokens: List[int]) -> str:
        # Mock detokenization  
        return f"decoded_{len(tokens)}_tokens"


class LanguageProcessor:
    """Processes natural language using RWKV model."""
    
    def __init__(self, model_path: Optional[str] = None, 
                 tokenizer_path: Optional[str] = None,
                 strategy: str = 'cpu fp32'):
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or os.path.join(
            os.path.dirname(__file__), '..', '20B_tokenizer.json'
        )
        self.strategy = strategy
        
        self.model = None
        self.pipeline = None
        self.is_loaded = False
        
        # Generation parameters
        self.default_params = {
            'temperature': 1.0,
            'top_p': 0.8,
            'max_tokens': 150,
            'stop_sequences': ['\n\n', 'Human:', 'Assistant:']
        }
        
        # Load model if path provided
        if model_path:
            self.load_model()
    
    def load_model(self) -> bool:
        """Load RWKV model and pipeline."""
        if not self.model_path:
            print("No model path provided")
            return False
        
        try:
            if RWKV_AVAILABLE:
                print(f"Loading RWKV model from {self.model_path}")
                self.model = RWKV(model=self.model_path, strategy=self.strategy)
                self.pipeline = PIPELINE(self.model, self.tokenizer_path)
                print("RWKV model loaded successfully")
            else:
                print("Using mock RWKV implementation")
                self.model = MockRWKV(self.model_path, self.strategy)
                self.pipeline = MockPipeline(self.model, self.tokenizer_path)
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Failed to load RWKV model: {e}")
            self.is_loaded = False
            return False
    
    def generate_response(self, prompt: str, **kwargs) -> RWKVResponse:
        """Generate a response using the RWKV model."""
        start_time = time.time()
        
        if not self.is_loaded:
            return RWKVResponse(
                text="Model not loaded",
                tokens=[],
                confidence=0.0,
                processing_time=0.0
            )
        
        # Merge parameters
        params = self.default_params.copy()
        params.update(kwargs)
        
        try:
            # Generate response
            response_text = self.pipeline.generate(
                prompt,
                token_count=params['max_tokens'],
                temperature=params['temperature'],
                top_p=params['top_p']
            )
            
            # Stop at stop sequences
            for stop_seq in params['stop_sequences']:
                if stop_seq in response_text:
                    response_text = response_text[:response_text.index(stop_seq)]
                    break
            
            # Tokenize response
            response_tokens = self.pipeline.encode(response_text)
            
            # Calculate confidence based on response length and parameters
            # Confidence heuristic: longer responses (up to 50 chars) indicate higher confidence
            # Modulated by temperature (higher temp = more uncertainty)
            MIN_RESPONSE_LENGTH = 50.0
            confidence = min(1.0, len(response_text) / MIN_RESPONSE_LENGTH / params['temperature'])
            
            processing_time = time.time() - start_time
            
            return RWKVResponse(
                text=response_text.strip(),
                tokens=response_tokens,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return RWKVResponse(
                text="Error generating response",
                tokens=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    def embed_text(self, text: str) -> List[float]:
        """Get text embedding from model (simplified)."""
        if not self.is_loaded:
            return [0.0] * 512  # Return zero embedding
        
        try:
            # Import torch here to handle cases where it might not be available
            import torch
            # Simplified embedding: use model's internal representations
            tokens = self.pipeline.encode(text)
            # In real implementation, would extract hidden states
            # For now, return mock embedding based on tokens
            embedding = [float(sum(tokens)) / len(tokens) if tokens else 0.0] * 512
            return embedding
        except ImportError:
            print("Torch not available, using zero embedding")
            return [0.0] * 512
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return [0.0] * 512


class RWKVCognitiveProcess(CognitiveProcess):
    """Cognitive process that integrates RWKV language processing."""
    
    def __init__(self, language_processor: LanguageProcessor):
        super().__init__("rwkv_processing", priority=0.8)
        self.language_processor = language_processor
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10
    
    def can_execute(self, architecture: 'RWKVCognitiveEngine') -> bool:
        return len(architecture.language_queries) > 0
    
    def _do_execute(self, architecture: 'RWKVCognitiveEngine') -> Dict[str, Any]:
        """Process language queries using RWKV."""
        responses_generated = 0
        
        while architecture.language_queries and responses_generated < 3:
            query = architecture.language_queries.pop(0)
            
            # Build prompt with context
            prompt = self._build_prompt(query, architecture)
            
            # Generate response
            response = self.language_processor.generate_response(prompt)
            
            # Add to conversation history
            self.conversation_history.append({
                'query': query,
                'response': response.text,
                'timestamp': time.time()
            })
            
            # Limit history size
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            # Convert response to knowledge atoms
            self._process_response(response, architecture)
            
            # Add response to output
            architecture.responses.append({
                'query': query,
                'response': response.text,
                'confidence': response.confidence,
                'processing_time': response.processing_time
            })
            
            responses_generated += 1
        
        return {'responses_generated': responses_generated}
    
    def _build_prompt(self, query: str, architecture: 'RWKVCognitiveEngine') -> str:
        """Build prompt with context from AtomSpace."""
        
        # Get relevant knowledge from AtomSpace
        relevant_concepts = []
        query_words = query.lower().split()
        
        for word in query_words[:5]:  # Limit context
            concepts = architecture.atomspace.get_atoms_by_name(word)
            relevant_concepts.extend(list(concepts)[:2])
        
        # Build context from conversation history
        context_lines = []
        for entry in self.conversation_history[-3:]:  # Last 3 exchanges
            context_lines.append(f"Human: {entry['query']}")
            context_lines.append(f"Assistant: {entry['response']}")
        
        # Build knowledge context
        knowledge_context = ""
        if relevant_concepts:
            knowledge_context = "\n\nRelevant knowledge:\n"
            for concept in relevant_concepts[:3]:
                if hasattr(concept, 'name') and concept.name:
                    knowledge_context += f"- {concept.name}\n"
        
        # Combine into final prompt
        prompt_parts = []
        
        if context_lines:
            prompt_parts.append("Previous conversation:\n" + "\n".join(context_lines))
        
        if knowledge_context:
            prompt_parts.append(knowledge_context)
        
        prompt_parts.append(f"\nHuman: {query}\nAssistant:")
        
        return "\n\n".join(prompt_parts)
    
    def _process_response(self, response: RWKVResponse, architecture: 'RWKVCognitiveEngine'):
        """Process RWKV response and extract knowledge."""
        
        # Extract key concepts from response
        response_words = response.text.lower().split()
        
        for word in response_words:
            if len(word) > 3 and word.isalpha():  # Filter meaningful words
                # Create or update concept node
                concept = architecture.atomspace.create_node(
                    AtomType.CONCEPT_NODE, word,
                    TruthValue(0.6, response.confidence)
                )
                architecture.agent.attention.focus_on(concept, 0.4)
        
        # Create response atom
        response_node = architecture.atomspace.create_node(
            AtomType.CONCEPT_NODE, f"response_{len(self.conversation_history)}",
            TruthValue(response.confidence, 0.8)
        )
        
        architecture.agent.attention.focus_on(response_node, 0.6)


class RWKVCognitiveEngine(CognitiveArchitecture):
    """
    Cognitive engine that integrates RWKV language model with OpenCog reasoning.
    Provides autonomous agentic inference capabilities.
    """
    
    def __init__(self, agent_name: str = "RWKVCogAgent", 
                 model_path: Optional[str] = None,
                 strategy: str = 'cpu fp32'):
        
        super().__init__(agent_name)
        
        # Language processing components
        self.language_processor = LanguageProcessor(model_path, strategy=strategy)
        
        # RWKV-specific queues
        self.language_queries: List[str] = []
        self.responses: List[Dict[str, Any]] = []
        
        # Add RWKV cognitive process
        self.processes['rwkv_processing'] = RWKVCognitiveProcess(self.language_processor)
        
        # Enhanced actions for language interaction
        self._register_rwkv_actions()
        
        # Conversation state
        self.conversation_state = {
            'active': False,
            'participant': None,
            'context': {}
        }
    
    def _register_rwkv_actions(self):
        """Register RWKV-specific actions."""
        
        def answer_question_action(agent):
            """Action to answer questions using RWKV."""
            if self.language_queries:
                query = self.language_queries[0]  # Process first query
                return f"Processing question: {query}"
            return "No questions to process"
        
        def explain_reasoning_action(agent):
            """Action to explain reasoning process."""
            focused_atoms = agent.attention.get_focused_atoms()
            if focused_atoms:
                concepts = [atom.name for atom in focused_atoms[:5] if atom.name]
                if concepts:
                    explanation_query = f"Explain the relationship between: {', '.join(concepts)}"
                    self.language_queries.append(explanation_query)
                    return f"Generating explanation for concepts: {concepts}"
            return "No concepts to explain"
        
        def generate_insight_action(agent):
            """Action to generate insights from knowledge."""
            # Get recent memories
            recent_episodes = agent.memory.recall_episodes({}, limit=5)
            if recent_episodes:
                context = [ep['event'] for ep in recent_episodes]
                insight_query = f"What insights can be drawn from these recent experiences: {context}"
                self.language_queries.append(insight_query)
                return "Generating insights from recent experiences"
            return "No experiences to analyze"
        
        # Register actions
        self.agent.register_action("answer_question", answer_question_action,
                                 "Answer questions using RWKV language model")
        self.agent.register_action("explain_reasoning", explain_reasoning_action,
                                 "Explain reasoning process in natural language")
        self.agent.register_action("generate_insight", generate_insight_action,
                                 "Generate insights from accumulated knowledge")
    
    def ask_question(self, question: str) -> str:
        """Ask a question to the cognitive engine."""
        self.language_queries.append(question)
        
        # Add input to processing buffer
        self.add_input(question)
        
        # Create a goal to answer the question
        goal_id = self.add_goal(f"Answer question: {question}", priority=0.8)
        
        return goal_id
    
    def start_conversation(self, participant: str = "Human"):
        """Start a conversation session."""
        self.conversation_state['active'] = True
        self.conversation_state['participant'] = participant
        self.conversation_state['context'] = {
            'start_time': time.time(),
            'turn_count': 0
        }
        
        # Add conversation goal
        self.add_goal("Engage in meaningful conversation", priority=0.7)
        
        print(f"Started conversation with {participant}")
    
    def end_conversation(self):
        """End the conversation session."""
        if self.conversation_state['active']:
            self.conversation_state['active'] = False
            duration = time.time() - self.conversation_state['context']['start_time']
            turn_count = self.conversation_state['context']['turn_count']
            
            # Add conversation summary to memory
            self.agent.memory.add_episode({
                'type': 'conversation_ended',
                'participant': self.conversation_state['participant'],
                'duration': duration,
                'turn_count': turn_count
            })
            
            print(f"Conversation ended. Duration: {duration:.1f}s, Turns: {turn_count}")
    
    def get_response(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Get the next response from the engine."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.responses:
                response = self.responses.pop(0)
                if self.conversation_state['active']:
                    self.conversation_state['context']['turn_count'] += 1
                return response
            
            time.sleep(0.1)
        
        return None
    
    def load_model(self, model_path: str, strategy: str = 'cpu fp32') -> bool:
        """Load RWKV model."""
        self.language_processor.model_path = model_path
        self.language_processor.strategy = strategy
        return self.language_processor.load_model()
    
    def add_conversational_knowledge(self, knowledge: str):
        """Add knowledge in conversational format."""
        # Process as both input and knowledge
        self.add_input(knowledge)
        self.add_knowledge(knowledge)
        
        # Also generate insights about the knowledge
        insight_query = f"What are the key insights from this knowledge: {knowledge}"
        self.language_queries.append(insight_query)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        rwkv_process = self.processes.get('rwkv_processing')
        
        if isinstance(rwkv_process, RWKVCognitiveProcess):
            return {
                'active': self.conversation_state['active'],
                'participant': self.conversation_state['participant'],
                'history_length': len(rwkv_process.conversation_history),
                'pending_queries': len(self.language_queries),
                'pending_responses': len(self.responses),
                'context': self.conversation_state['context']
            }
        
        return {'error': 'RWKV process not available'}
    
    def export_conversation(self) -> List[Dict[str, Any]]:
        """Export conversation history."""
        rwkv_process = self.processes.get('rwkv_processing')
        
        if isinstance(rwkv_process, RWKVCognitiveProcess):
            return rwkv_process.conversation_history.copy()
        
        return []
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get enhanced status including RWKV-specific information."""
        base_status = self.get_status()
        
        rwkv_status = {
            'language_model': {
                'loaded': self.language_processor.is_loaded,
                'model_path': self.language_processor.model_path,
                'strategy': self.language_processor.strategy
            },
            'conversation': self.get_conversation_summary(),
            'language_processing': {
                'pending_queries': len(self.language_queries),
                'pending_responses': len(self.responses)
            }
        }
        
        base_status.update(rwkv_status)
        return base_status
    
    def __str__(self):
        model_status = "loaded" if self.language_processor.is_loaded else "not loaded"
        return f"RWKVCognitiveEngine('{self.agent.name}', model: {model_status})"