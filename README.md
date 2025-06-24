# Self-Reflection LLM

A sophisticated conversational AI system that implements the Reflexion framework for autonomous agents with self-improvement capabilities. The system can operate in both conversational mode and autonomous agent mode, learning from failures through trajectory summarization, evaluation, and self-reflection.

## < Key Features

- **Dual-Mode Operation**: Seamlessly transitions between conversational assistant and autonomous agent
- **Self-Reflection Framework**: Implements the Reflexion methodology for learning from failures
- **Multi-Model Architecture**: Uses specialized models for different tasks (planning, evaluation, reflection)
- **Dynamic Tool Integration**: MCP-based tool loading with shell, web search, TTS, and more
- **Iterative Improvement**: Learns from failed attempts and refines strategies automatically

## <× Architecture

### Agent States
- **PLAN**: Generate comprehensive step-by-step plans
- **PRE_AGENT_LOOP**: Initialize and validate agent tasks
- **AGENT_LOOP**: Execute tasks systematically with available tools
- **EXIT_AGENT_LOOP**: Evaluate results and trigger self-reflection if needed

### Self-Reflection Process
1. **Trajectory Summarization**: Condenses agent actions into structured summaries
2. **Binary Evaluation**: Pass/fail assessment of task completion
3. **Self-Reflection**: Generates actionable feedback for failed attempts
4. **Iterative Retry**: Uses reflection feedback to improve next attempts

### Model Specialization
- **Actor Models**: GPT-4.1, O3/O4 series for planning and execution
- **Evaluator Models**: O4-mini, O3-mini for trajectory assessment
- **Self-Reflection Models**: O3, O4-mini, O3-mini for generating improvement feedback
- **Summarizer Models**: GPT-4.1 series for trajectory condensation

## =à Available Tools

### Core Tools
- **Planning**: `generate_plan`, `move_to_next_step`
- **Execution**: `think`, `start_agent_loop`, `exit_agent_loop`
- **Shell**: Command execution via Claude MCP
- **Web Search**: Brave Search and Exa Search integration
- **Text-to-Speech**: ElevenLabs TTS capabilities
- **Utilities**: Time and timezone functions

## =Ë Requirements

- Python 3.12+
- OpenAI API key
- Optional: API keys for integrated services (Brave Search, ElevenLabs, Exa)

## =€ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd self_reflection_llm
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## <¯ Usage

### Command Line Interface

```bash
# Basic usage
self-reflection-llm -p mcp_servers.json -a gpt-4.1 -e o4-mini -sr o3 -sm gpt-4.1-mini

# With custom page token size
self-reflection-llm -p mcp_servers.json -a gpt-4.1 -e o4-mini -sr o3 -sm gpt-4.1-mini -ptz 256000
```

### Parameters

- `-p, --path2mcp_settings`: Path to MCP servers configuration file
- `-a, --actor_model`: Model for planning and execution
- `-e, --evaluator_model`: Model for trajectory evaluation
- `-sr, --self_reflection_model`: Model for generating self-reflection feedback
- `-sm, --summary_model`: Model for trajectory summarization
- `-ptz, --page_token_size`: Token limit for trajectory pages (default: 128000)

### Example Session

```
> Hello! Can you help me analyze the latest AI research papers?

# System transitions to conversational mode
I'd be happy to help you analyze AI research papers! To do this effectively, I can:
- Search for recent papers using web search
- Analyze and summarize findings
- Compare different approaches

Would you like me to start working on this systematically?

> Yes, please focus on self-reflection and autonomous agents

# System transitions to agent mode
[Agent begins systematic planning and execution...]
```

## ™ Configuration

### MCP Servers Configuration

The `mcp_servers.json` file defines available tools and services:

```json
{
  "mcpServers": [
    {
      "name": "Shell",
      "mcp_config": {
        "command": "claude",
        "args": ["mcp", "serve"]
      },
      "include_tools": ["Bash"]
    },
    {
      "name": "BraveWebSearch",
      "mcp_config": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {
          "BRAVE_API_KEY": "your_api_key_here"
        }
      },
      "include_tools": ["*"]
    }
  ]
}
```

### Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key
BRAVE_API_KEY=your_brave_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

## =, Research Background

This implementation is based on the Reflexion framework, which enables language model agents to learn from task feedback through:

- **Verbal Reinforcement**: Self-generated reflections rather than gradient updates
- **Trial and Error**: Iterative task attempts with progressive improvement
- **Memory Augmentation**: Persistent reflection history across attempts

## > Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## =Ä License

This project is licensed under the MIT License - see the LICENSE file for details.

## =O Acknowledgments

- Based on the Reflexion framework for self-reflective language agents
- Uses OpenAI's latest models including O3 and O4 series
- Integrates with MCP (Model Context Protocol) for tool management

## =Ú References

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)