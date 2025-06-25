"""
LLM Engine module for handling conversational AI interactions with OpenAI models.

This module provides the LLMEngine class which manages conversations with OpenAI's language models,
handles tool calls, and processes responses. It supports both standard chat completions and 
streaming responses with tool calling capabilities.

The engine can work with different OpenAI models and supports parallel tool execution.
It implements an async context manager pattern for proper resource management.

Classes:
    LLMEngine: Main class for managing LLM conversations and tool interactions

Dependencies:
    - asyncio: For asynchronous operations
    - zmq: For message queuing
    - openai: For interacting with OpenAI API
    - pydantic: For data validation
"""

import click 
import yaml 
import asyncio 

import json 
import zmq 
import zmq.asyncio as aiozmq 

from typing import List, Tuple, Dict, Any, Optional, Set, Coroutine 
from typing import AsyncGenerator, Self 

from contextlib import asynccontextmanager, AsyncExitStack, suppress

from abc import ABC, abstractmethod 
from operator import itemgetter, attrgetter 
from functools import partial, reduce 

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion, ChatCompletionChunk, ParsedChatCompletion,
    ChatCompletionMessageToolCall
)

from pydantic import BaseModel 
from enum import Enum 
from uuid import uuid4


from .types import ChatMessage, Role, StopReason, LLMModels, OpenaiModel
from .system_instructions import SystemInstructions
from .settings.credentials import Credentials
from .log import logger 
from .mcp_router import MCPRouter
from .mcp_types import McpSettings, ToolSchema, McpConfig
from .tools_definitions import TOOLS

from .system_instructions import SystemInstructions
from .reflexion import Reflexion

class AgentState(str, Enum):
    PLAN = "plan"
    PRE_AGENT_LOOP = "pre_agent_loop"
    AGENT_LOOP = "agent_loop"
    EXIT_AGENT_LOOP = "exit_agent_loop"

class LLMEngine:
    """
    A class to manage conversations with OpenAI language models and handle tool interactions.

    This class provides functionality to:
    - Maintain conversation state
    - Generate responses from OpenAI models
    - Handle tool calls and their responses
    - Process streaming responses
    - Manage async resources properly

    Attributes:
        credentials (Credentials): API credentials for OpenAI
        tool_calls_handler (ToolCallsHandler): Handler for processing tool calls
        models (LLMModels): Configuration for which models to use
        openai_client (AsyncOpenAI): Async client for OpenAI API
        ctx (aiozmq.Context): ZMQ context for async operations

    Methods:
        run_loop: Main conversation loop
        generate_response: Generate response from OpenAI
        consume_response: Process streaming response
        handle_tool_calls: Execute tool calls
        apply_tool_call: Execute single tool call
    """

    def __init__(
            self, 
            credentials:Credentials, 
            mcp_settings:List[McpSettings], 
            models:LLMModels, 
            page_token_size:int=128000, 
            max_iterations:int=3,
            ):
        self.credentials = credentials 
        self.mcp_settings = mcp_settings 
        self.models = models 
        self.reflexion = Reflexion(
            credentials=self.credentials,
            summary_model=self.models.summarizer,
            evaluation_model=self.models.evaluator,
            self_reflection_model=self.models.self_reflection
        )
        self.page_token_size = page_token_size
        self.max_iterations = max_iterations 
        
        self.state = AgentState.PRE_AGENT_LOOP
        self.tool_choice:str = "auto"  # can be auto | required | none 
        
        self.task:str = None # store the query of the user 
        self.iteration_count:int = 0 
        self.reflexion_feedbacks:List[str] = []
        
        
    async def __aenter__(self) -> Self:
        self.openai_client = AsyncOpenAI(api_key=self.credentials.OPENAI_API_KEY)
        self.ctx = aiozmq.Context()
        self.resource_manager = AsyncExitStack()
        self.mcp_router = MCPRouter(mcp_settings=self.mcp_settings)
        self.mcp_router = await self.resource_manager.enter_async_context(self.mcp_router)
        await self.mcp_router.start_all_mcp_servers()
        mcp_tools = self.mcp_router.list_all_tools()
        
        TOOLS["think"]["function"]["parameters"]["properties"]["next_tool"]["properties"]["name"]["enum"].extend(
            [tool.name for tool in mcp_tools]
        )
        actions = []
        for tool in mcp_tools:
            actions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        

        self.agent_state2tools_hmap:Dict[AgentState, List[Dict[str, Any]]] = {
            AgentState.PRE_AGENT_LOOP: [TOOLS['start_agent_loop']],
            AgentState.PLAN: [TOOLS['generate_plan']],
            AgentState.AGENT_LOOP: [TOOLS['exit_agent_loop'], TOOLS["think"], TOOLS['move_to_next_step'], *actions],
            AgentState.EXIT_AGENT_LOOP: [], # no tools to call in this state
        }

        return self 
    
    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            logger.error(f"Error in run_loop: {exc_type} {exc_value}")
            logger.exception(traceback)
        self.ctx.term()
        await self.resource_manager.aclose()
    
    async def run_loop(self, max_tokens:int=1024, reasoning_effort:Optional[str]=None, parallel_tool_calls:bool=False) -> None:
        if self.models.actor in [OpenaiModel.O4_MINI, OpenaiModel.O3_MINI, OpenaiModel.O3]:
            if reasoning_effort is None:
                raise ValueError("reasoning_effort is required for O4_MINI, O3_MINI, O3")
            assert reasoning_effort in ["low", "medium", "high"], "reasoning_effort must be one of low, medium, high"

        conversation:List[ChatMessage] = []
        stop_reason = StopReason.STOP 
        while True:
            try:
                if stop_reason != StopReason.TOOL_CALLS:
                    match self.state:
                        case AgentState.EXIT_AGENT_LOOP:  
                            # eval the trajectory and check if a new iteration is needed or not 
                            evaluation_score, evaluation_reasoning, reflexion_feedbacks_updated = await self.reflexion.process(
                                task=self.task, conversation=conversation, 
                                page_token_size=self.page_token_size, reflexion_feedbacks=self.reflexion_feedbacks
                            )  
                            feedback_signal = self.update_state(
                                evaluation_score, evaluation_reasoning, reflexion_feedbacks_updated
                            )
                            if feedback_signal is None:
                                logger.info('agent was able to fully complete the task')
                                continue  # new loop with a clean agent state => user will take the turn
                            logger.info('a new iteration will start with reflection feedback')
                            query = feedback_signal
                        case _:
                            query = input('> ')  # user prompt 
                    
                    self.conversation.append(ChatMessage(role=Role.USER, content=query))
                response = await self.generate_response(
                    max_tokens=max_tokens, 
                    reasoning_effort=reasoning_effort, 
                    parallel_tool_calls=parallel_tool_calls, 
                    conversation=conversation
                )
                delta_stop_reason, delta_conversation = await self.consume_response(response)
                stop_reason = delta_stop_reason 
                conversation.extend(delta_conversation)
                print('\n')
            except asyncio.CancelledError:
                break 
            except Exception as e:
                logger.error(f"Error in run_loop: {e}")
                break 
    
    def reset_agent_state(self) -> None:
        self.state = AgentState.PRE_AGENT_LOOP
        self.task = None
        self.iteration_count = 0

    def update_state(self, evaluation_score:bool, evaluation_reasoning:str, reflexion_feedbacks_updated:List[str]) -> Optional[str]:
        # current agent state is exit_agent_loop
        # this function will set the agent state to either pre_agent_loop(success) or plan(failure)
        if evaluation_score:  # True => agent has passed the task
            self.reset_agent_state()  # agent is ready for a new task 
            return None 
        
        if self.iteration_count == self.max_iterations:
            logger.warning(f'agent was not able to solve this task after {self.max_iterations} attempts')
            self.reset_agent_state()
            return None 
        # start again the agent loop without new task or confirmation... jump to generate plan step(this is like a backtrack)
        self.reflexion_feedbacks = reflexion_feedbacks_updated
        self.state = AgentState.PLAN  # activate the `plan -> agent_loop` sequence again
        self.tool_choice = "required"
        return yaml.dump({
            "trajectory_scoring": {
                "pass/fail": "failed",
                "reason": evaluation_reasoning
            },
            "feedback_signal": self.reflexion_feedbacks[-1], 
            "tips": "check your previous plan, if needed, you can update it with a backtrack. this is optional",
            "message": "please, in this new iteration, try to correct your trajectory by taking into account the given feedback"
        })

    async def generate_response(
            self, 
            max_tokens:int=1024, 
            reasoning_effort:Optional[str]=None, 
            parallel_tool_calls:bool=False, 
            conversation:List[ChatMessage]=[]
            ) -> AsyncGenerator[ChatCompletionChunk, None]:
        

        match self.state:
            case AgentState.PLAN | AgentState.AGENT_LOOP:
                system_prompt = SystemInstructions.ACTOR_PROMPT.value 
            case _: 
                system_prompt = SystemInstructions.CONVERSATIONAL_PROMPT.value.format(
                    services=", ".join(self.mcp_router.list_all_services())
                )

        system_message = [
            ChatMessage(
                role=Role.SYSTEM,
                content=system_prompt
            )
        ]

        tools = self.agent_state2tools_hmap[self.state]
                
        if self.models.actor in [OpenaiModel.O4_MINI, OpenaiModel.O3_MINI, OpenaiModel.O3]:
            partial_completion = partial(
                self.openai_client.chat.completions.create, 
                model=self.models.actor, 
                max_completion_tokens=max_tokens, 
                stream=True, 
                reasoning_effort=reasoning_effort, 
                tools=tools,
                tool_choice=self.tool_choice
            )
        else:
            partial_completion = partial(
                self.openai_client.chat.completions.create, 
                model=self.models.actor, 
                max_completion_tokens=max_tokens, 
                stream=True, 
                tools=tools, 
                parallel_tool_calls=parallel_tool_calls,
                tool_choice=self.tool_choice
            )
        response:AsyncGenerator[ChatCompletionChunk, None] = await partial_completion(
            messages=system_message + conversation,
        )  
        return response 

    async def consume_response(self, response:AsyncGenerator[ChatCompletionChunk, None]) -> Tuple[StopReason, List[ChatMessage]]:
        content = ''
        tool_call_accumulator:Dict[int, ChatCompletionMessageToolCall] = {}
        stop_reason = StopReason.STOP 
        conversation:List[ChatMessage] = []
        async for chunk in response:
            if chunk.choices[0].finish_reason is not None:
                stop_reason = chunk.choices[0].finish_reason
                tool_calls = None 
                if len(tool_call_accumulator) > 0:
                    tool_calls = [tool_call.model_dump() for tool_call in tool_call_accumulator.values()]
                    tool_call_accumulator.clear()
                    logger.info(f'number of tool calls to make: {len(tool_calls)}')

                assistant_message = ChatMessage(
                    role=Role.ASSISTANT,
                    content=content,
                    tool_calls=tool_calls
                )
                conversation.append(assistant_message)
                match stop_reason:
                    case StopReason.TOOL_CALLS:
                        tool_calls_result = await self.handle_tool_calls(tool_calls)
                        conversation.extend(tool_calls_result)
                    case _:
                        break 
            
            delta_content = chunk.choices[0].delta.content or ''
            content = content + delta_content
            print(click.style(delta_content, fg='blue', bg='white', bold=True), end='', flush=True)
            if chunk.choices[0].delta.tool_calls is None:
                continue

            for tool_call in chunk.choices[0].delta.tool_calls:
                if not tool_call.index in tool_call_accumulator:
                    tool_call_accumulator[tool_call.index] = tool_call
                tool_call_accumulator[tool_call.index].function.arguments += tool_call.function.arguments
        return stop_reason, conversation 
    
    async def handle_tool_calls(self, tool_calls:List[Dict[str, Any]]) -> List[ChatMessage]:
        tool_calls_results = await asyncio.gather(*[self.apply_tool_call(tool_call) for tool_call in tool_calls])
        return tool_calls_results 

    async def apply_tool_call(self, tool_call:Dict[str, Any]) -> ChatMessage:
        try:
            function_name = tool_call['function']['name']
            function_arguments = json.loads(tool_call['function']['arguments'])
            print(click.style('--------------------------------', fg='green'))
            print(click.style(function_name, fg='green'))
            print(click.style(tool_call['id'], fg='green'))
            print(json.dumps(function_arguments, indent=2, default=str))
            print(click.style('--------------------------------', fg='green'))

            match function_name:
                case 'generate_plan' | 'start_agent_loop' | 'exit_agent_loop' | 'move_to_next_step' | 'think':
                    fn_ = attrgetter(function_name)(self) 
                    tool_call_result_content = await fn_(**function_arguments)
                case _:  # calling an mcp tool (dynamic mapping) can scale to a vast number of mcp servers
                    # due to the mcp router capabilities (multiple mcp servers mounted)
                    tool_call_result_content = await self.mcp_router.higher_order_apply(
                        tool_call_name=function_name,
                        tool_call_arguments=function_arguments
                    )

            return ChatMessage(
                role=Role.TOOL,
                content=tool_call_result_content,
                tool_call_id=tool_call['id']
            )
        except Exception as e:
            logger.error(f"Error in apply_tool_call: {e}")
            return ChatMessage(
                role=Role.TOOL,
                content=f"Error in apply_tool_call: {e}",
                tool_call_id=tool_call['id']
            )
    

    async def start_agent_loop(self, task:str, confirmation:str) -> str:
        self.state = AgentState.PLAN  # next action should be plan generation 
        self.tool_choice = "required"  # tool call is mandatory
        self.task = task # keep the task of the user as query for evaluation part 
        return yaml.dump({
            "setup": {
                "task": task,
                "confirmation": confirmation
            },
            "message": "the agent loop was started successfully, please generate a plan"
        }, sort_keys=False)
    
    async def generate_plan(self, steps:List[str], justification:str) -> str:
        self.state = AgentState.AGENT_LOOP  # starting the agent_loop(exit_agent_loop, think, move_to_next_step, *action)
        return yaml.dump({
            "generated_plan": {
                "steps": steps,
                "justification": justification
            },
            "message": "the plan was generated successfully, please execute the plan, always think before you act. once you complete a step, move to the next step"
        }, sort_keys=False)

    async def think(self, thought:str, next_tool:Dict[str, Any]) -> str:
        print(thought)
        return yaml.dump({
            "next_tool_to_call": next_tool,
            "message": "the next tool to call was selected successfully, please call the tool with appropriate arguments."
        }, sort_keys=False)
    
    async def move_to_next_step(self, completed_step:str, next_step:str) -> str:
        return yaml.dump({
            "completed_step": completed_step,
            "next_step": next_step,
            "message": f"the next step {next_step} was selected successfully, please execute the next step"
        }, sort_keys=False)

    async def exit_agent_loop(self, reason:str) -> str:
        print(reason)   
        self.state = AgentState.EXIT_AGENT_LOOP  # no more tool call
        self.tool_choice = None # force the LLM to generate a final response 
        self.iteration_count += 1 # iteration completed 
        return yaml.dump({
            "iterations_left": self.max_iterations - self.iteration_count,
            "context": "each agent loop is an iteration, in case of failure, the next iteration will be called with some user feedback",  
            "end_of_agent_loop": "the agent loop was stopped, please generate the final result if you succeeded else, tell the user the reason why you were not able to build the result"
        })
        
    
    