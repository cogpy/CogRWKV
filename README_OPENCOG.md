# OpenCog Integration with ChatRWKV

This implementation provides an OpenCog-inspired autonomous agentic inference engine integrated with the RWKV language model. It combines symbolic reasoning capabilities with neural language processing to create intelligent, goal-directed autonomous agents.

## Features

### 🧠 AtomSpace Knowledge Representation
- **Hypergraph Structure**: Stores knowledge as atoms (nodes and links) in a hypergraph
- **Truth Values**: Each atom has associated strength and confidence values
- **Type System**: Supports various atom types (ConceptNode, PredicateNode, InheritanceLink, etc.)
- **Thread-Safe**: Concurrent access support with locking mechanisms

### 🔍 Pattern Matching and Reasoning
- **Query Engine**: Sophisticated pattern matching for knowledge retrieval
- **Variable Binding**: Support for variables in patterns with constraint checking
- **Similarity Search**: Find similar atoms based on structure and content
- **Inference**: Basic reasoning capabilities including inheritance and similarity

### 🤖 Autonomous Agents
- **Goal-Directed Behavior**: Agents pursue goals with priority-based scheduling
- **Action System**: Extensible action framework with built-in cognitive actions
- **Memory Systems**: Episodic, semantic, and working memory integration
- **Attention Mechanism**: Focus-based attention allocation with decay

### 🏗️ Cognitive Architecture
- **Multi-Process System**: Concurrent cognitive processes (perception, reasoning, planning, learning, metacognition)
- **Cognitive States**: Dynamic state tracking (arousal, valence, cognitive load, confidence)
- **Process Orchestration**: Priority-based process execution with resource management
- **Introspection**: Self-monitoring and adaptive behavior

### 🗣️ RWKV Language Integration
- **Neural-Symbolic Bridge**: Seamless integration between symbolic reasoning and neural language processing
- **Conversational AI**: Natural language interaction with context awareness
- **Knowledge Extraction**: Convert language responses into structured knowledge atoms
- **Contextual Generation**: Use AtomSpace knowledge to inform language generation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                RWKVCognitiveEngine                      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Language   │  │   Cognitive  │  │   Autonomous  │  │
│  │ Processor   │  │ Architecture │  │     Agent     │  │
│  │   (RWKV)    │  │              │  │               │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ AtomSpace   │  │   Pattern    │  │   Memory &    │  │
│  │ Knowledge   │  │   Matcher    │  │  Attention    │  │
│  │ Repository  │  │              │  │               │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from opencog import RWKVCognitiveEngine

# Create cognitive engine
engine = RWKVCognitiveEngine("MyAgent")

# Add knowledge
engine.add_knowledge("Cats are mammals")
engine.add_knowledge("Mammals are animals") 

# Start the engine
engine.start()

# Ask questions
goal_id = engine.ask_question("What are cats?")

# Get response
response = engine.get_response(timeout=5.0)
print(f"Response: {response['response']}")

# Stop engine
engine.stop()
```

### With RWKV Model

```python
from opencog import RWKVCognitiveEngine

# Create engine with RWKV model
engine = RWKVCognitiveEngine(
    "SmartAgent",
    model_path="/path/to/rwkv-model.pth",
    strategy="cuda fp16"  # or "cpu fp32"
)

# Load the model
if engine.load_model():
    print("Model loaded successfully")
    
    # Start conversation
    engine.start_conversation("User")
    engine.start()
    
    # Interactive loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        engine.ask_question(user_input)
        response = engine.get_response(timeout=10.0)
        
        if response:
            print(f"Agent: {response['response']}")
        else:
            print("Agent: No response")
    
    engine.stop()
    engine.end_conversation()
```

### Advanced Knowledge Integration

```python
from opencog import AtomSpace, AtomType, TruthValue, Pattern, Query

# Create knowledge base
atomspace = AtomSpace()

# Add structured knowledge
cat = atomspace.create_node(AtomType.CONCEPT_NODE, "cat", TruthValue(0.9, 0.8))
animal = atomspace.create_node(AtomType.CONCEPT_NODE, "animal", TruthValue(0.95, 0.9))

# Create relationships
inheritance = atomspace.create_link(
    AtomType.INHERITANCE_LINK, [cat, animal], TruthValue(0.9, 0.9)
)

# Query knowledge
from opencog import PatternMatcher
matcher = PatternMatcher(atomspace)

pattern = Pattern(
    AtomType.INHERITANCE_LINK,
    outgoing=[
        Pattern(AtomType.CONCEPT_NODE, "$child"),
        Pattern(AtomType.CONCEPT_NODE, "animal")
    ]
)

query = Query(pattern)
results = matcher.execute_query(query)

for result in results:
    child = result.bindings.get('child')
    print(f"{child.name} is an animal")
```

## Core Components

### AtomSpace
- **Purpose**: Central knowledge repository using hypergraph representation
- **Key Features**: Thread-safe, type-indexed, exportable/importable
- **Usage**: Store and manage all symbolic knowledge

### PatternMatcher  
- **Purpose**: Query engine for knowledge retrieval and reasoning
- **Key Features**: Variable binding, constraint checking, similarity search
- **Usage**: Find patterns, make inferences, discover relationships

### AutonomousAgent
- **Purpose**: Goal-directed autonomous behavior
- **Key Features**: Goal management, action execution, memory systems
- **Usage**: Create intelligent agents with autonomous behavior

### CognitiveArchitecture
- **Purpose**: High-level cognitive system orchestration
- **Key Features**: Multi-process execution, state management, introspection
- **Usage**: Coordinate complex cognitive behaviors

### RWKVCognitiveEngine
- **Purpose**: Integration layer for RWKV language models
- **Key Features**: Neural-symbolic bridge, conversational AI, knowledge extraction
- **Usage**: Create language-capable autonomous agents

## Cognitive Processes

### Perception Process
- Processes inputs from environment
- Converts raw data to structured atoms
- Updates attention and knowledge base

### Reasoning Process  
- Performs inference over knowledge base
- Applies inheritance and similarity reasoning
- Discovers implicit knowledge relationships

### Planning Process
- Manages goals and creates action plans
- Schedules actions based on goal priorities
- Monitors goal progress and completion

### Learning Process
- Strengthens frequently used knowledge
- Creates associations between co-occurring concepts
- Updates truth values based on experience

### Metacognition Process
- Monitors cognitive state and performance
- Adjusts cognitive parameters dynamically
- Determines appropriate cognitive modes

## Configuration

### Model Configuration
```python
# CPU usage
engine = RWKVCognitiveEngine(strategy="cpu fp32")

# GPU usage
engine = RWKVCognitiveEngine(strategy="cuda fp16")

# Multi-GPU
engine = RWKVCognitiveEngine(strategy="cuda:0 fp16 -> cuda:1 fp16")
```

### Cognitive Parameters
```python
# Adjust cognitive cycle timing
architecture.cycle_interval = 0.5  # seconds

# Configure attention system
architecture.agent.attention.max_focus_size = 15

# Set generation parameters
architecture.language_processor.default_params.update({
    'temperature': 0.8,
    'top_p': 0.9,
    'max_tokens': 200
})
```

## Examples

See `opencog_demo.py` for comprehensive examples covering:
- Basic AtomSpace operations
- Pattern matching and queries  
- Autonomous agent behavior
- Cognitive architecture integration
- RWKV language model integration
- Knowledge integration workflows

## Requirements

### Core Requirements
- Python 3.8+
- PyTorch 1.13+
- tokenizers
- prompt_toolkit

### Optional (for full RWKV integration)
- RWKV pip package
- CUDA toolkit (for GPU acceleration)

## Installation

1. Install dependencies:
```bash
pip install torch tokenizers prompt_toolkit
```

2. Optional RWKV package:
```bash
pip install rwkv
```

3. Import and use:
```python
from opencog import RWKVCognitiveEngine
```

## Performance Considerations

- **AtomSpace Size**: Performance degrades with very large knowledge bases (>100K atoms)
- **Attention Focus**: Limit focused atoms to 20-50 for optimal performance
- **Cognitive Cycles**: Balance cycle frequency with computational load
- **Memory Management**: Periodic cleanup of episodic memory recommended

## Limitations

- **Model Dependency**: Full functionality requires RWKV model files
- **Single Agent**: Current implementation supports one primary agent per engine
- **Language Support**: Optimized for English, other languages may need tuning
- **Scalability**: Designed for research/prototype scale, not production deployment

## Future Development

- **Multi-Agent Systems**: Support for multiple interacting agents
- **Distributed AtomSpace**: Scale to larger knowledge bases
- **Enhanced Reasoning**: More sophisticated inference mechanisms  
- **Tool Integration**: API calls and external tool usage
- **Persistent Storage**: Database backend for large-scale deployment

## Contributing

The OpenCog integration is designed to be extensible:

1. **Custom Cognitive Processes**: Inherit from `CognitiveProcess`
2. **New Atom Types**: Extend the `AtomType` class
3. **Action Extensions**: Register custom actions with agents
4. **Pattern Extensions**: Create specialized pattern matching logic

## License

This implementation follows the same license as the parent ChatRWKV project.