#!/usr/bin/env python3
"""
OpenCog-RWKV Integration Demo
Demonstrates the autonomous agentic inference engine with RWKV language model.
"""

import os
import sys
import time
import json
from typing import Dict, List, Any

# Add opencog module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'opencog'))

from opencog import (
    AtomSpace, Atom, Node, Link, AtomType, TruthValue,
    PatternMatcher, Query, Pattern,
    AutonomousAgent, Goal, Action, GoalStatus,
    CognitiveArchitecture,
    RWKVCognitiveEngine
)


def demo_basic_atomspace():
    """Demonstrate basic AtomSpace functionality."""
    print("\n" + "="*60)
    print("DEMO 1: Basic AtomSpace Operations")
    print("="*60)
    
    # Create AtomSpace
    atomspace = AtomSpace()
    
    # Create some nodes
    cat = atomspace.create_node(AtomType.CONCEPT_NODE, "cat", TruthValue(0.9, 0.8))
    animal = atomspace.create_node(AtomType.CONCEPT_NODE, "animal", TruthValue(0.95, 0.9))
    mammal = atomspace.create_node(AtomType.CONCEPT_NODE, "mammal", TruthValue(0.9, 0.85))
    
    print(f"Created nodes:")
    print(f"  {cat}")
    print(f"  {animal}")
    print(f"  {mammal}")
    
    # Create inheritance relationships
    cat_is_animal = atomspace.create_link(
        AtomType.INHERITANCE_LINK, [cat, animal], TruthValue(0.9, 0.9)
    )
    cat_is_mammal = atomspace.create_link(
        AtomType.INHERITANCE_LINK, [cat, mammal], TruthValue(0.95, 0.9)
    )
    mammal_is_animal = atomspace.create_link(
        AtomType.INHERITANCE_LINK, [mammal, animal], TruthValue(0.98, 0.95)
    )
    
    print(f"\nCreated inheritance links:")
    print(f"  {cat_is_animal}")
    print(f"  {cat_is_mammal}")
    print(f"  {mammal_is_animal}")
    
    print(f"\nAtomSpace statistics: {atomspace.get_statistics()}")
    
    return atomspace


def demo_pattern_matching(atomspace: AtomSpace):
    """Demonstrate pattern matching capabilities."""
    print("\n" + "="*60)
    print("DEMO 2: Pattern Matching and Queries")
    print("="*60)
    
    # Create pattern matcher
    matcher = PatternMatcher(atomspace)
    
    # Query 1: Find all concepts that inherit from "animal"
    print("\nQuery 1: Find all concepts that inherit from 'animal'")
    pattern = Pattern(
        AtomType.INHERITANCE_LINK,
        outgoing=[
            Pattern(AtomType.CONCEPT_NODE, "$child"),
            Pattern(AtomType.CONCEPT_NODE, "animal")
        ]
    )
    query = Query(pattern, limit=10)
    results = matcher.execute_query(query)
    
    print(f"Found {len(results)} results:")
    for result in results:
        child_concept = result.bindings.get('child')
        if child_concept:
            print(f"  - {child_concept.name} is an animal (score: {result.score:.3f})")
    
    # Query 2: Find similar atoms to "cat"
    cat_node = atomspace.get_atoms_by_name("cat")
    if cat_node:
        cat = list(cat_node)[0]
        print(f"\nQuery 2: Find atoms similar to 'cat'")
        similar = matcher.find_similar_atoms(cat, similarity_threshold=0.5)
        print(f"Found {len(similar)} similar atoms:")
        for atom in similar:
            similarity = matcher._calculate_similarity(cat, atom)
            print(f"  - {atom.name} (similarity: {similarity:.3f})")


def demo_autonomous_agent():
    """Demonstrate autonomous agent functionality."""
    print("\n" + "="*60)
    print("DEMO 3: Autonomous Agent")
    print("="*60)
    
    # Create agent
    agent = AutonomousAgent("DemoAgent")
    
    # Add some knowledge
    agent.add_knowledge("learning")
    agent.add_knowledge("reasoning")
    agent.add_knowledge("problem_solving")
    
    print(f"Created agent: {agent}")
    print(f"Agent knowledge size: {agent.atomspace.size()}")
    
    # Add goals
    learning_goal = Goal(
        description="Learn new concepts and relationships",
        priority=0.8,
        deadline=time.time() + 30  # 30 seconds from now
    )
    
    exploration_goal = Goal(
        description="Explore knowledge connections",
        priority=0.6,
        deadline=time.time() + 45
    )
    
    agent.add_goal(learning_goal)
    agent.add_goal(exploration_goal)
    
    print(f"\nAdded goals:")
    print(f"  - {learning_goal}")
    print(f"  - {exploration_goal}")
    
    # Start agent and let it run for a short while
    print("\nStarting agent for 5 seconds...")
    agent.start()
    time.sleep(5)
    agent.stop()
    
    # Check status
    status = agent.get_status()
    print(f"\nAgent final status:")
    print(f"  Statistics: {status['statistics']}")
    print(f"  Goals: {len(status['goals'])} goals")
    print(f"  Knowledge: {status['knowledge_size']} atoms")
    
    return agent


def demo_cognitive_architecture():
    """Demonstrate cognitive architecture."""
    print("\n" + "="*60)
    print("DEMO 4: Cognitive Architecture")
    print("="*60)
    
    # Create cognitive architecture
    cog_arch = CognitiveArchitecture("CogDemo")
    
    print(f"Created cognitive architecture: {cog_arch}")
    
    # Add some inputs
    cog_arch.add_input("The cat is sleeping on the mat")
    cog_arch.add_input("Cats are mammals")
    cog_arch.add_input({"animal_type": "feline", "behavior": "hunting"})
    
    # Add goals
    goal1_id = cog_arch.add_goal("Process incoming information", priority=0.9)
    goal2_id = cog_arch.add_goal("Learn from examples", priority=0.7)
    
    print(f"Added goals: {goal1_id}, {goal2_id}")
    
    # Run for a short period
    print("\nStarting cognitive architecture for 8 seconds...")
    cog_arch.start()
    
    # Monitor for a few cycles
    for i in range(4):
        time.sleep(2)
        status = cog_arch.get_status()
        print(f"  Cycle {i+1}: Mode={status['state']['mode']}, "
              f"Load={status['state']['cognitive_load']:.2f}, "
              f"Processes={len([p for p in status['processes'].values() if p['active']])}")
    
    cog_arch.stop()
    
    # Get final status
    final_status = cog_arch.get_status()
    print(f"\nFinal cognitive architecture status:")
    print(f"  Total cycles: {final_status['statistics']['cognitive_cycles']}")
    print(f"  Average cycle duration: {final_status['statistics']['average_cycle_duration']:.3f}s")
    print(f"  Agent knowledge: {final_status['agent_status']['knowledge_size']} atoms")
    
    return cog_arch


def demo_rwkv_integration():
    """Demonstrate RWKV cognitive engine (without actual model)."""
    print("\n" + "="*60)
    print("DEMO 5: RWKV Cognitive Engine")
    print("="*60)
    
    # Create RWKV cognitive engine (will use mock implementation)
    rwkv_engine = RWKVCognitiveEngine("RWKVDemo")
    
    print(f"Created RWKV engine: {rwkv_engine}")
    
    # Add some conversational knowledge
    rwkv_engine.add_conversational_knowledge("Artificial intelligence is the simulation of human intelligence")
    rwkv_engine.add_conversational_knowledge("Machine learning is a subset of AI")
    rwkv_engine.add_conversational_knowledge("Deep learning uses neural networks")
    
    # Start conversation
    rwkv_engine.start_conversation("DemoUser")
    
    # Start the engine
    print("\nStarting RWKV cognitive engine...")
    rwkv_engine.start()
    
    # Ask some questions
    questions = [
        "What is artificial intelligence?",
        "How does machine learning relate to AI?",
        "What are the benefits of deep learning?"
    ]
    
    for question in questions:
        print(f"\nAsking: {question}")
        goal_id = rwkv_engine.ask_question(question)
        
        # Wait for response
        response = rwkv_engine.get_response(timeout=3.0)
        if response:
            print(f"Response: {response['response']}")
            print(f"Confidence: {response['confidence']:.3f}")
        else:
            print("No response received")
        
        time.sleep(1)
    
    # Get conversation summary
    summary = rwkv_engine.get_conversation_summary()
    print(f"\nConversation summary: {summary}")
    
    # Stop engine
    rwkv_engine.stop()
    rwkv_engine.end_conversation()
    
    # Get enhanced status
    status = rwkv_engine.get_enhanced_status()
    print(f"\nRWKV Engine final status:")
    print(f"  Model loaded: {status['language_model']['loaded']}")
    print(f"  Total cognitive cycles: {status['statistics']['cognitive_cycles']}")
    print(f"  Knowledge atoms: {status['agent_status']['knowledge_size']}")
    
    return rwkv_engine


def demo_knowledge_integration():
    """Demonstrate integration of different knowledge types."""
    print("\n" + "="*60)
    print("DEMO 6: Knowledge Integration")
    print("="*60)
    
    # Create engine
    engine = RWKVCognitiveEngine("KnowledgeDemo")
    
    # Add structured knowledge
    atomspace = engine.atomspace
    
    # Create a small knowledge base about animals
    animals = ["cat", "dog", "bird", "fish"]
    properties = ["mammal", "vertebrate", "pet", "wild"]
    
    for animal in animals:
        animal_node = atomspace.create_node(
            AtomType.CONCEPT_NODE, animal, TruthValue(0.9, 0.8)
        )
        
        # Add some properties
        if animal in ["cat", "dog"]:
            mammal_node = atomspace.create_node(
                AtomType.CONCEPT_NODE, "mammal", TruthValue(0.95, 0.9)
            )
            pet_node = atomspace.create_node(
                AtomType.CONCEPT_NODE, "pet", TruthValue(0.8, 0.7)
            )
            
            atomspace.create_link(
                AtomType.INHERITANCE_LINK, [animal_node, mammal_node],
                TruthValue(0.95, 0.9)
            )
            atomspace.create_link(
                AtomType.INHERITANCE_LINK, [animal_node, pet_node],
                TruthValue(0.8, 0.8)
            )
    
    print(f"Created knowledge base with {atomspace.size()} atoms")
    
    # Start engine and add contextual knowledge
    engine.start()
    
    # Add contextual information
    contexts = [
        "Pets are domesticated animals that live with humans",
        "Mammals are warm-blooded vertebrates that feed milk to their young",
        "Cats and dogs are popular household pets"
    ]
    
    for context in contexts:
        engine.add_input(context)
        time.sleep(0.5)
    
    # Let it process for a while
    time.sleep(3)
    
    # Query the integrated knowledge
    print("\nQuerying integrated knowledge...")
    
    # Create pattern to find pet mammals
    pattern = Pattern(
        AtomType.CONCEPT_NODE, "$animal"
    )
    
    matcher = PatternMatcher(atomspace)
    query = Query(pattern, limit=10)
    results = matcher.execute_query(query)
    
    print(f"Found {len(results)} concepts:")
    for result in results:
        animal = result.bindings.get('animal')
        if animal and animal.name:
            related = matcher.get_related_atoms(animal, max_depth=1)
            print(f"  {animal.name}: connected to {len(related)} other concepts")
    
    engine.stop()
    
    # Export knowledge
    knowledge_export = atomspace.export_to_dict()
    print(f"\nKnowledge export contains {len(knowledge_export['atoms'])} atoms")
    
    return engine


def main():
    """Run all demonstrations."""
    print("OpenCog-RWKV Integration Demonstration")
    print("=====================================")
    print("This demo shows the integration of OpenCog-inspired")
    print("cognitive architecture with RWKV language models.")
    
    try:
        # Run demonstrations
        atomspace = demo_basic_atomspace()
        demo_pattern_matching(atomspace)
        demo_autonomous_agent()
        demo_cognitive_architecture()
        demo_rwkv_integration()
        demo_knowledge_integration()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("All OpenCog-RWKV integration features demonstrated successfully!")
        print("\nKey capabilities shown:")
        print("  ✓ AtomSpace knowledge representation")
        print("  ✓ Pattern matching and reasoning")
        print("  ✓ Autonomous agent behavior")
        print("  ✓ Cognitive architecture integration")
        print("  ✓ RWKV language model integration")
        print("  ✓ Multi-modal knowledge processing")
        
        print("\nTo use with a real RWKV model:")
        print("  1. Download an RWKV model file")
        print("  2. Update model_path in RWKVCognitiveEngine")
        print("  3. Install RWKV pip package if not available")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()