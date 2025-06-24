from typing import Dict

TOOLS: Dict[str, Dict] = {
    "start_agent_loop": {
        "type": "function",
        "function": {
            "name": "start_agent_loop",
            "description": "Initializes the agent execution loop and validates the user's request. This is the entry point for all agent interactions and must be the first tool called before any other operations. The agent operates within a service-oriented architecture where specialized capabilities are dynamically loaded as needed through service clients.",
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmation": {
                        "type": "string",
                        "description": "The confirmation for starting the agent loop given by the user",
                    }, 
                    "task": {
                        "type": "string",
                        "description": "Detailed description of the task to be performed by the agent",
                    }
                },
                "required": ["confirmation", "task"],
            },
        },
    },
    "generate_plan": {
        "type": "function",
        "function": {
            "name": "generate_plan",
            "description": (
                "Create a high‑level roadmap (a checklist) that decomposes a complex question or task into discrete, logical steps.\n\n"
                "**WHEN TO CALL**: Always as the *first* action after reading a new user request.\n\n"
                "**WHY**: • Establishes clear sub‑goals and prevents overlooking essential research facets.\n"
                "        • Serves as a progress tracker—other tools reference these step labels.\n\n"
                "**HOW**: Provide the `steps` parameter as a newline‑separated checklist, phrased imperatively (e.g., 'Identify relevant statutes').\n"
                "The function returns an internal representation; no direct user output is produced—you must later narrate your progress."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": "Ordered list of sub‑tasks, each on its own line. The wording SHOULD match the legal research workflow (e.g., 'Search Code civil for definitions', 'Analyse jurisprudence').",
                        "items": {
                            "type": "string",
                        }
                    }, 
                    "justification": {
                        "type": "string",
                        "description": "A justification for the steps. Why did you choose these steps?",
                    }
                },
                "required": ["steps", "justification"],
            },
        },
    },
    "move_to_next_step": {
        "type": "function",
        "function": {
            "name": "move_to_next_step",
            "description": (
                "Mark a checklist item as complete **and** declare the next active step.\n\n"
                "**WHEN TO CALL**: Immediately after finishing *any* step in your roadmap.\n\n"
                "**WHY**: Keeps the internal plan in sync with actual progress, enabling transparent, linear reasoning and facilitating mid‑course corrections.\n\n"
                "**HOW**:\n  • `completed_step`: Paste the exact text of the finished checklist entry.\n  • `next_step`: Paste the *exact* text of the upcoming checklist entry.  If the plan is finished, set `next_step` to an empty string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "completed_step": {
                        "type": "string",
                        "description": "The precise checklist label just accomplished",
                    },
                    "next_step": {
                        "type": "string",
                        "description": "The precise checklist label to work on next (or empty if plan completed)",
                    },
                },
                "required": ["completed_step", "next_step"],
            },
        },
    },
    "think": {
        "type": "function",
        "function": {
            "name": "think",
            "description": (
                "Record internal deliberations (ephemeral notes) without affecting external state.\n\n"
                "**WHEN TO CALL**: Before *any* tool invocation that requires non‑trivial reasoning—especially\n\n"
                "**WHY**: Captures chain‑of‑thought for auditability, allowing future self_reflect analysis.\n\n"
                "**HOW**: Pass a free‑form string summarising the rationale. *Do not* include sensitive user data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "The private reasoning to log",
                    }, 
                    "next_tool": {
                        "type": "object",
                        "description": "The next tool to call",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the tool to call",
                                "enum": ["exit_agent_loop", "move_to_next_step"]
                            }, 
                            "justification": {
                                "type": "string",
                                "description": "The justification for the tool to call. Why did you choose this tool? if you decide to interact_with_environment, specify which subtool to load.",
                            }
                        },
                        "required": ["name", "justification"],
                    }
                },
                "required": ["thought", "next_tool"],
            },
        },
    },
    "exit_agent_loop": {
        "type": "function",
        "function": {
            "name": "exit_agent_loop",
            "description": (
                "Terminate the current agent processing loop gracefully.\n\n"
                "**WHEN TO CALL**:\n  • Task is complete and all required information has been provided.\n  • A critical error occurs that prevents further progress.\n  • The question is beyond scope or ethically inappropriate.\n  • User explicitly requests to stop processing.\n\n"
                "**WHY**: • Prevents unnecessary computation.\n"
                "        • Allows for clean termination rather than timing out.\n"
                "        • Creates clear boundaries around task completion.\n\n"
                "**HOW**: Provide a concise reason for ending the loop. This will be logged internally but not directly shown to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "The reason for ending the agent loop",
                    }
                },
                "required": ["reason"],
            },
        },
    }
}
