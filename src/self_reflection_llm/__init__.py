import click 
import json 
import asyncio 

from .llm_engine import LLMEngine
from .settings.credentials import Credentials
from .mcp_types import McpSettings
from .types import LLMModels

from typing import Optional


@click.command()
@click.option("--path2mcp_settings", "-p" , type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--actor_model", "-a", type=str, required=True)
@click.option("--evaluator_model", "-e", type=str, required=True)
@click.option("--self_reflection_model", "-sr", type=str, required=True)
@click.option("--summary_model", "-sm", type=str, required=True)
@click.option("--page_token_size", "-ptz", type=int, default=128_000)
@click.option("--max_tokens", "-mt", type=int, default=4096)
@click.option("--reasoning_effort", "-re", type=click.Choice(["low", "medium", "high"]), default=None)
@click.option("--parallel_tool_calls", "-prl", type=bool, default=False)
def main(
    path2mcp_settings:str, 
    actor_model:str, 
    evaluator_model:str, 
    self_reflection_model:str, 
    summary_model:str, 
    page_token_size:int,
    max_tokens:int=4096,
    reasoning_effort:Optional[str]=None,
    parallel_tool_calls:bool=False,
    ) -> None:
    credentials = Credentials()
    models = LLMModels(
        actor=actor_model,
        evaluator=evaluator_model,
        self_reflection=self_reflection_model,
        summary=summary_model
    )
    with open(path2mcp_settings, "r") as f:
        mcp_settings = json.load(f)
    mcp_settings = [McpSettings(**mcp_setting) for mcp_setting in mcp_settings["mcpServers"]]

    llm_engine = LLMEngine(
        credentials=credentials, 
        mcp_settings=mcp_settings, 
        models=models, 
        page_token_size=page_token_size
    )

    async def main_loop() -> None:
        async with llm_engine:
            await llm_engine.run_loop(
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                parallel_tool_calls=parallel_tool_calls
            )

    asyncio.run(main_loop())
