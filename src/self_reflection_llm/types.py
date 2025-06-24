from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional, Any, Union, get_origin, get_args
from enum import Enum 

class Role(str, Enum):
    SYSTEM: str = 'system'
    USER: str = 'user'
    ASSISTANT: str = 'assistant'
    DEVELOPER: str = 'developer'
    TOOL:str = 'tool'

class ChatMessage(BaseModel):
    role:Role 
    content: str | List[Dict[str, Any]] | None = None 
    tool_call_id: str | None = None
    tool_calls: List[Dict[str, Any]] | None = None 

class StopReason(str, Enum):
    STOP : str = 'stop'
    TOOL_CALLS : str = 'tool_calls'
    LENGTH : str = 'length'

class OpenaiModel(str, Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_41 = "gpt-4.1"
    GPT_41_MINI = "gpt-4.1-mini"
    GPT_41_NANO = "gpt-4.1-nano"
    O4_MINI = "o4-mini"
    O3_MINI = "o3-mini"
    O3 = "o3"

class LLMModels(BaseModel):
    actor:OpenaiModel=OpenaiModel.GPT_41
    evaluator:OpenaiModel=OpenaiModel.O4_MINI
    self_reflection:OpenaiModel=OpenaiModel.O3
    summarizer:OpenaiModel=OpenaiModel.GPT_41_MINI