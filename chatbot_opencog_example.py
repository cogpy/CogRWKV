#!/usr/bin/env python3
"""
ChatBot OpenCog Integration Example
Shows how to integrate OpenCog cognitive architecture with existing ChatRWKV infrastructure.
"""

import os
import sys
import time
from typing import Dict, List, Any

# Add paths for both opencog and existing ChatRWKV modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'opencog'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from opencog import (
    RWKVCognitiveEngine, AtomSpace, AtomType, TruthValue,
    AutonomousAgent, Goal, Pattern, Query
)

# Try to import existing ChatRWKV utilities
try:
    from utils import TOKENIZER
    CHATRWKV_AVAILABLE = True
except ImportError:
    CHATRWKV_AVAILABLE = False
    print("ChatRWKV utilities not available, using OpenCog-only mode")


class OpenCogChatBot:
    """
    Intelligent chatbot that combines OpenCog reasoning with RWKV language generation.
    Integrates with existing ChatRWKV infrastructure while adding autonomous cognitive capabilities.
    """
    
    def __init__(self, name: str = "OpenCogBot", model_path: str = None):
        self.name = name
        
        # Initialize cognitive engine
        self.cognitive_engine = RWKVCognitiveEngine(name, model_path)
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if CHATRWKV_AVAILABLE:
            try:
                tokenizer_path = os.path.join(os.path.dirname(__file__), "20B_tokenizer.json")
                if os.path.exists(tokenizer_path):
                    self.tokenizer = TOKENIZER(tokenizer_path)
                    print(f"✓ Loaded ChatRWKV tokenizer from {tokenizer_path}")
            except Exception as e:
                print(f"Could not load tokenizer: {e}")
        
        # Conversation state
        self.conversation_active = False
        self.conversation_history = []
        
        # Initialize knowledge base with common conversational knowledge
        self._initialize_knowledge_base()
        
        # Set up cognitive goals
        self._setup_cognitive_goals()
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with common conversational knowledge."""
        
        # Add basic conversational concepts
        knowledge_items = [
            "conversation is communication between people",
            "questions require answers",
            "politeness improves communication", 
            "context helps understanding",
            "learning improves performance",
            "memory stores information",
            "attention focuses on important things",
            "reasoning helps solve problems",
            "language expresses thoughts",
            "intelligence involves learning and adaptation"
        ]
        
        for knowledge in knowledge_items:
            self.cognitive_engine.add_knowledge(knowledge)
        
        # Create some structured relationships
        atomspace = self.cognitive_engine.atomspace
        
        # Communication concepts
        communication = atomspace.create_node(
            AtomType.CONCEPT_NODE, "communication", TruthValue(0.95, 0.9)
        )
        conversation = atomspace.create_node(
            AtomType.CONCEPT_NODE, "conversation", TruthValue(0.9, 0.85)
        )
        language = atomspace.create_node(
            AtomType.CONCEPT_NODE, "language", TruthValue(0.95, 0.9)
        )
        
        # Create inheritance relationships
        atomspace.create_link(
            AtomType.INHERITANCE_LINK, [conversation, communication],
            TruthValue(0.9, 0.9)
        )
        atomspace.create_link(
            AtomType.INHERITANCE_LINK, [language, communication], 
            TruthValue(0.85, 0.8)
        )
        
        print(f"✓ Initialized knowledge base with {atomspace.size()} atoms")
    
    def _setup_cognitive_goals(self):
        """Set up high-level cognitive goals for the chatbot."""
        
        goals = [
            {
                'description': "Engage in meaningful conversation",
                'priority': 0.9
            },
            {
                'description': "Learn from user interactions", 
                'priority': 0.7
            },
            {
                'description': "Maintain conversational context",
                'priority': 0.8
            },
            {
                'description': "Provide helpful responses",
                'priority': 0.9
            }
        ]
        
        for goal_info in goals:
            goal_id = self.cognitive_engine.add_goal(
                goal_info['description'],
                priority=goal_info['priority']
            )
            print(f"✓ Added goal: {goal_info['description']} (ID: {goal_id})")
    
    def start(self):
        """Start the chatbot and cognitive engine."""
        print(f"\n🤖 Starting {self.name}...")
        
        # Start cognitive engine
        self.cognitive_engine.start()
        
        # Start conversation mode
        self.cognitive_engine.start_conversation("User")
        self.conversation_active = True
        
        print(f"✓ {self.name} is ready for conversation!")
        print("💡 The bot has autonomous cognitive capabilities:")
        print("   - Goal-directed behavior")  
        print("   - Memory and learning")
        print("   - Attention and focus")
        print("   - Reasoning and inference")
        print("   - Context awareness")
    
    def stop(self):
        """Stop the chatbot and cognitive engine."""
        if self.conversation_active:
            self.cognitive_engine.end_conversation()
            self.conversation_active = False
        
        self.cognitive_engine.stop()
        print(f"🤖 {self.name} stopped.")
    
    def chat(self, user_input: str) -> str:
        """Process user input and generate response."""
        if not self.conversation_active:
            return "Bot is not active. Please start the conversation first."
        
        # Record user input
        self.conversation_history.append({
            'speaker': 'user',
            'message': user_input,
            'timestamp': time.time()
        })
        
        # Process input through cognitive engine
        goal_id = self.cognitive_engine.ask_question(user_input)
        
        # Get response from cognitive engine
        response_data = self.cognitive_engine.get_response(timeout=5.0)
        
        if response_data and response_data['response']:
            bot_response = response_data['response']
            confidence = response_data.get('confidence', 0.5)
        else:
            # Fallback response if cognitive engine doesn't respond
            bot_response = self._generate_fallback_response(user_input)
            confidence = 0.3
        
        # Record bot response
        self.conversation_history.append({
            'speaker': 'bot',
            'message': bot_response,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Learn from interaction
        self._learn_from_interaction(user_input, bot_response)
        
        return bot_response
    
    def _generate_fallback_response(self, user_input: str) -> str:
        """Generate fallback response when cognitive engine doesn't respond."""
        
        # Simple pattern-based fallback responses
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm an OpenCog-powered chatbot. How can I help you?"
        
        elif any(word in input_lower for word in ['how', 'what', 'why', 'when', 'where']):
            return "That's an interesting question. Let me think about that using my reasoning capabilities."
        
        elif any(word in input_lower for word in ['thank', 'thanks']):
            return "You're welcome! I'm glad I could help."
        
        elif any(word in input_lower for word in ['bye', 'goodbye', 'exit']):
            return "Goodbye! It was nice chatting with you."
        
        else:
            return "I'm processing your input through my cognitive architecture. Could you tell me more about what you're interested in?"
    
    def _learn_from_interaction(self, user_input: str, bot_response: str):
        """Learn from the interaction by updating knowledge and attention."""
        
        # Extract key concepts from user input
        input_words = user_input.lower().split()
        
        # Focus attention on key concepts
        for word in input_words:
            if len(word) > 3 and word.isalpha():
                concept = self.cognitive_engine.atomspace.create_node(
                    AtomType.CONCEPT_NODE, word, TruthValue(0.7, 0.6)
                )
                self.cognitive_engine.agent.attention.focus_on(concept, 0.5)
        
        # Add conversational knowledge
        interaction_knowledge = f"User said '{user_input}' and bot responded '{bot_response}'"
        self.cognitive_engine.add_conversational_knowledge(interaction_knowledge)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the chatbot."""
        
        status = {
            'name': self.name,
            'conversation_active': self.conversation_active,
            'conversation_length': len(self.conversation_history),
            'cognitive_status': self.cognitive_engine.get_enhanced_status()
        }
        
        # Add conversation summary
        if self.conversation_history:
            recent_messages = self.conversation_history[-5:]  # Last 5 messages
            status['recent_conversation'] = [
                f"{msg['speaker']}: {msg['message'][:50]}..."
                for msg in recent_messages
            ]
        
        return status
    
    def analyze_conversation(self) -> Dict[str, Any]:
        """Analyze the conversation using cognitive capabilities."""
        
        if not self.conversation_history:
            return {'error': 'No conversation to analyze'}
        
        # Basic conversation analysis
        user_messages = [msg for msg in self.conversation_history if msg['speaker'] == 'user']
        bot_messages = [msg for msg in self.conversation_history if msg['speaker'] == 'bot']
        
        # Calculate average confidence
        confidences = [msg.get('confidence', 0.5) for msg in bot_messages if 'confidence' in msg]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Extract topics (simple word frequency)
        all_words = []
        for msg in user_messages:
            all_words.extend(msg['message'].lower().split())
        
        word_freq = {}
        for word in all_words:
            if len(word) > 3 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        top_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_exchanges': len(user_messages),
            'average_confidence': avg_confidence,
            'top_topics': top_topics,
            'conversation_duration': (
                self.conversation_history[-1]['timestamp'] - 
                self.conversation_history[0]['timestamp']
            ) if len(self.conversation_history) > 1 else 0,
            'cognitive_insights': self.cognitive_engine.get_conversation_summary()
        }


def interactive_chat_demo():
    """Run an interactive chat demonstration."""
    
    print("🧠 OpenCog-RWKV Chatbot Demo")
    print("=" * 50)
    
    # Create and start chatbot
    chatbot = OpenCogChatBot("CognitiveBot")
    chatbot.start()
    
    print("\n💬 Chat with the bot (type 'quit' to exit, 'status' for status, 'analyze' for analysis)")
    print("-" * 50)
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            elif user_input.lower() == 'status':
                # Show status
                status = chatbot.get_status()
                print(f"\n📊 Bot Status:")
                print(f"   Conversation length: {status['conversation_length']} exchanges")
                print(f"   Knowledge atoms: {status['cognitive_status']['agent_status']['knowledge_size']}")
                print(f"   Cognitive cycles: {status['cognitive_status']['statistics']['cognitive_cycles']}")
                print(f"   Active goals: {len(status['cognitive_status']['agent_status']['goals'])}")
                continue
            
            elif user_input.lower() == 'analyze':
                # Show conversation analysis
                analysis = chatbot.analyze_conversation()
                if 'error' not in analysis:
                    print(f"\n🔍 Conversation Analysis:")
                    print(f"   Total exchanges: {analysis['total_exchanges']}")
                    print(f"   Average confidence: {analysis['average_confidence']:.2f}")
                    print(f"   Duration: {analysis['conversation_duration']:.1f} seconds")
                    print(f"   Top topics: {[topic[0] for topic in analysis['top_topics'][:3]]}")
                else:
                    print(f"\n❌ {analysis['error']}")
                continue
            
            # Process normal chat input
            print("🤔 (thinking...)")
            response = chatbot.chat(user_input)
            print(f"🤖 Bot: {response}")
    
    except KeyboardInterrupt:
        print("\n\n⚡ Chat interrupted by user")
    
    finally:
        # Clean shutdown
        print("\n🛑 Shutting down chatbot...")
        chatbot.stop()
        
        # Show final analysis
        print("\n📋 Final Analysis:")
        analysis = chatbot.analyze_conversation()
        if 'error' not in analysis:
            print(f"   Total conversation exchanges: {analysis['total_exchanges']}")
            print(f"   Average response confidence: {analysis['average_confidence']:.2f}")
            print(f"   Main conversation topics: {[t[0] for t in analysis['top_topics'][:3]]}")
        
        print("\n✅ Demo completed successfully!")


def automated_demo():
    """Run an automated demonstration with scripted interactions."""
    
    print("\n🚀 Automated OpenCog-RWKV Demo")
    print("=" * 40)
    
    # Create chatbot
    chatbot = OpenCogChatBot("AutoDemo")
    chatbot.start()
    
    # Scripted conversation
    test_inputs = [
        "Hello! How are you?",
        "What do you know about artificial intelligence?",
        "Can you learn from our conversation?",
        "What is the purpose of reasoning?",
        "How does memory work in cognitive systems?",
        "Thank you for the interesting conversation!"
    ]
    
    print("\n🎭 Running scripted conversation...")
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n[{i}/{len(test_inputs)}] User: {user_input}")
        
        # Add thinking delay for realism
        time.sleep(0.5)
        
        response = chatbot.chat(user_input)
        print(f"[{i}/{len(test_inputs)}] Bot: {response}")
        
        # Brief pause between exchanges
        time.sleep(1)
    
    # Show final status
    print("\n📊 Final Chatbot Status:")
    status = chatbot.get_status()
    print(f"   Knowledge base: {status['cognitive_status']['agent_status']['knowledge_size']} atoms")
    print(f"   Conversation length: {status['conversation_length']} exchanges") 
    print(f"   Cognitive cycles completed: {status['cognitive_status']['statistics']['cognitive_cycles']}")
    
    # Show analysis
    analysis = chatbot.analyze_conversation()
    if 'error' not in analysis:
        print(f"   Average confidence: {analysis['average_confidence']:.2f}")
        print(f"   Top topics discussed: {[t[0] for t in analysis['top_topics'][:3]]}")
    
    chatbot.stop()
    print("\n✅ Automated demo completed!")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_chat_demo()
    else:
        automated_demo()
        
        # Offer interactive mode
        response = input("\n🤔 Would you like to try interactive mode? (y/n): ").lower()
        if response in ['y', 'yes']:
            interactive_chat_demo()