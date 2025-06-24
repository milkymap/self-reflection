from pydantic import BaseModel
from typing import Dict, List, Any, Optional


class ToolSchema(BaseModel):
    name:str 
    description:str 
    parameters:Dict[str, Any]

class McpConfig(BaseModel):
    command:str
    args:List[str]
    env:Optional[Dict[str, str]]=None

class McpSettings(BaseModel):
    name:str
    mcp_config:McpConfig
    exclude_tools:List[str]=[]
    include_tools:List[str]=[]
    ignore:bool=False
    startup_timeout:float=30