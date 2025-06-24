import asyncio 
from itertools import chain 
from contextlib import AsyncExitStack
from typing import Dict, Any, List, Self 


from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from  .log import logger 

from .mcp_types import ToolSchema, McpSettings


class MCPRouter:
    def __init__(self, mcp_settings:List[McpSettings]): 
        self.mcp_settings = mcp_settings
        self.mcp_sessions:Dict[str, ClientSession] = {}
        self.mcp_tools:Dict[str, List[ToolSchema]] = {}
        self.async_exit_stack = AsyncExitStack()
        
    async def __aenter__(self) -> Self:
        self.mcp_sessions_mutex = asyncio.Lock()
        return self 
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(f"Error in MCPRouter: {exc_type} {exc_value}")
            logger.exception(traceback)
        
        await self.async_exit_stack.aclose()
    
    async def start_mcp_server(self, server_config:McpSettings) -> None:
        server_parameters = StdioServerParameters(
            command=server_config.mcp_config.command,
            args=server_config.mcp_config.args,
            env=server_config.mcp_config.env,
        )
        logger.info(f"Starting MCP server {server_config.name} with command {server_config.mcp_config.command}")
        cleint = stdio_client(server=server_parameters)
        stdio_transport = await self.async_exit_stack.enter_async_context(cleint)
        reader, writer = stdio_transport
        session_ = ClientSession(reader, writer)
        try:
            session = await self.async_exit_stack.enter_async_context(session_)
            async with asyncio.timeout(server_config.startup_timeout):
                await session.initialize()
        except TimeoutError:
            logger.error(f"MCP server {server_config.name} startup timeout")
            raise 

        logger.info(f"MCP server {server_config.name} initialized")
        list_tools_result = await session.list_tools()
        tools = [
            ToolSchema(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
            ) for tool in list_tools_result.tools
        ]
        logger.info(f'MCP server {server_config.name} has {len(tools)} tools')
        if len(server_config.exclude_tools) > 0:
            tools = [tool for tool in tools if tool.name not in server_config.exclude_tools]
        
        if len(server_config.include_tools) > 0:
            if len(server_config.include_tools) != 1 or server_config.include_tools[0] != '*':
                tools = [tool for tool in tools if tool.name in server_config.include_tools]
        
        if len(tools) == 0:
            logger.error(f"No tools found for server {server_config.name}")
            raise ValueError(f"No tools found for server {server_config.name}")
    
        for tool in tools:
            tool.name = f'mcp__{server_config.name}__{tool.name}'
        
        async with self.mcp_sessions_mutex: 
            self.mcp_sessions[server_config.name] = session
            self.mcp_tools[server_config.name] = tools

        logger.info(f"MCP server {server_config.name} started")
            
    async def start_all_mcp_servers(self):
        tasks = []
        for server_config in self.mcp_settings:
            if not server_config.ignore:
                task = asyncio.create_task(self.start_mcp_server(server_config))
                tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)
        if len(self.mcp_tools) == 0:
            raise ValueError("No MCP servers started")
        logger.info(f"All MCP servers started")
    
    def list_all_tools(self) -> List[ToolSchema]:
        return list(chain(*self.mcp_tools.values()))
    
    def list_all_services(self) -> List[str]:
        return list(self.mcp_tools.keys())
    
    async def higher_order_apply(self, tool_call_name:str, tool_call_arguments:Dict[str, Any]) -> str:
        _, server_name, tool_name = tool_call_name.split('__')
        if server_name not in self.mcp_sessions:
            raise ValueError(f"MCP server {server_name} not found")
        async with self.mcp_sessions_mutex:
            session_reference = self.mcp_sessions[server_name]
        result = await session_reference.call_tool(name=tool_name, arguments=tool_call_arguments)
        response = '\n'.join([ res.model_dump_json(indent=2) for res in result.content ])
        
        print(response)
        return response 
    
    