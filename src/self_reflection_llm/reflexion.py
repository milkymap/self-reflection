import asyncio
import yaml 

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ParsedChatCompletion

from .system_instructions import SystemInstructions
from .settings.credentials import Credentials
from .types import OpenaiModel, ChatMessage
from typing import List 

from tiktoken import encoding_for_model
from .log import logger

from pydantic import BaseModel, Field

SUMMARY_MODELS = [
    OpenaiModel.GPT_41_NANO,
    OpenaiModel.GPT_41_MINI, 
    OpenaiModel.GPT_4O_MINI, 
    OpenaiModel.GPT_4O,
    OpenaiModel.GPT_41
]

EVALUATION_MODELS = [
    OpenaiModel.O3_MINI,
    OpenaiModel.O4_MINI 
]

SELF_REFLECTION_MODELS = [
    OpenaiModel.O3_MINI,
    OpenaiModel.O4_MINI,
    OpenaiModel.O3 
]


class Reflexion:
    def __init__(self, 
                 credentials:Credentials, 
                 summary_model:OpenaiModel=OpenaiModel.GPT_41_MINI, 
                 evaluation_model:OpenaiModel=OpenaiModel.O4_MINI, 
                 self_reflection_model:OpenaiModel=OpenaiModel.O3,
                 ):
        
        assert summary_model in SUMMARY_MODELS, "summary_model must be one of " + ", ".join(SUMMARY_MODELS) 
        assert evaluation_model in EVALUATION_MODELS, "evaluation_model must be one of " + ", ".join(EVALUATION_MODELS)
        assert self_reflection_model in SELF_REFLECTION_MODELS, "self_reflection_model must be one of " + ", ".join(SELF_REFLECTION_MODELS)
        
        self.credentials = credentials
        self.summary_model = summary_model
        self.evaluation_model = evaluation_model
        self.self_reflection_model = self_reflection_model
        self.openai_client = AsyncOpenAI(api_key=self.credentials.OPENAI_API_KEY)

    def group_trajectory_by_page(self, trajectory:list[str], page_token_size:int=128_000) -> list[str]:
        tokenizer = encoding_for_model(model_name="gpt-4o")
        nb_tokens_per_step = [ len(tokenizer.encode(text=step)) for step in trajectory ]

        
        pages = []
        current_page = []
        current_page_tokens = 0
        
        for i, step_tokens in enumerate(nb_tokens_per_step):
            if current_page_tokens + step_tokens > page_token_size:
                pages.append("\n".join(current_page))
                current_page = [trajectory[i]]
                current_page_tokens = step_tokens
            else:
                current_page.append(trajectory[i])
                current_page_tokens += step_tokens
                
        if current_page:
            pages.append("\n".join(current_page))
        
        print("+++++Compactify+++++")
        print(f"Total tokens: {sum(nb_tokens_per_step)}")
        print(f"Number of pages: {len(pages)} with page token size = {page_token_size} tokens")
        return pages

    async def summarize_page(self, page: str) -> str:
        response = await self.openai_client.chat.completions.create(
            model=self.summary_model.value,
            messages=[
                {"role": "system", "content": SystemInstructions.TRAJECTORY_SUMMARIZATION_PROMPT.value},
                {"role": "user", "content": page}
            ],
            max_tokens=16_384,  # make it a variable for the next version
        )
        return response.choices[0].message.content

    async def summarize_trajectory(
            self,
            trajectory:list[str], 
            page_token_size:int=128_000,
            ) -> str:
        pages = self.group_trajectory_by_page(trajectory, page_token_size)
        summaries = await asyncio.gather(*[self.summarize_page(page) for page in pages])
        for page_id, summary in enumerate(summaries):
            print(f'summary of page {page_id + 1}')
            print(summary)
            print('-------------------------------')
    
        return "\n".join(summaries)


    async def evaluate_trajectory(
            self,
            query:str, 
            trajectory_summary:str, 
            final_res:str, 
            ) -> tuple[bool, str]:
        class Evaluation(BaseModel):
            score: bool = Field(description="The evaluation of the trajectory. true(pass) or false(fail).")
            reasoning:str = Field(description="The reasoning behind the evaluation")
        
        response:ParsedChatCompletion = await self.openai_client.beta.chat.completions.parse(
            model=self.evaluation_model.value,
            messages=[
                {
                    "role": "system",
                    "content": SystemInstructions.EVALUATOR_PROMPT.value
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Task: {query}"
                        },
                        {
                            "type": "text",
                            "text": f"Assistant Trajectory: {trajectory_summary}"
                        },
                        {
                            "type": "text",
                            "text": f"Final Result of the agent: {final_res}"
                        }
                    ]
                }
            ],
            max_completion_tokens=100_000,
            response_format=Evaluation,
            reasoning_effort="high",
        )

        parsed_response:Evaluation = response.choices[0].message.parsed
        return parsed_response.score, parsed_response.reasoning

    async def self_reflect(
            self,
            query:str, 
            trajectory_summary:str, 
            evaluation_score:str, 
            evaluation_reasoning:str, 
            previous_reflection:str, 
            ) -> str:
        
        response = await self.openai_client.chat.completions.create(
            model=self.self_reflection_model.value,
            messages=[
                {
                    "role": "system",
                    "content": SystemInstructions.SELF_REFLECTION_PROMPT.value
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Task: {query}"
                        },
                        {
                            "type": "text",
                            "text": f"Last Trajectory Summary: {trajectory_summary}"
                        },
                        {
                            "type": "text",
                            "text": f"Evaluation Score: {evaluation_score}"
                        },
                        {
                            "type": "text",
                            "text": f"Evaluation Reasoning: {evaluation_reasoning}"
                        },
                        {
                            "type": "text",
                            "text": f"Previous Reflections: {previous_reflection}"
                        }
                    ]
                }
            ],
            max_completion_tokens=100_000,
            reasoning_effort="high",
        )
        
        return response.choices[0].message.content
    
    async def process(self, task:str, conversation:List[ChatMessage], page_token_size:int, reflexion_feedbacks:List[str]): 
        trajectory = []
        for msg in conversation:
            trajectory.append(msg.model_dump_json(indent=3))

        final_res = trajectory[-1]
        print(f'final result of the agent: {final_res}')
        trajectory_summary = await self.summarize_trajectory(
            trajectory=trajectory, 
            page_token_size=page_token_size
        )
        evaluation_score, evaluation_reasoning = await self.evaluate_trajectory(
            query=task, 
            trajectory_summary=trajectory_summary, 
            final_res=final_res
        )
        
        print(yaml.dump(
            {
                "score": evaluation_score,
                "reason": evaluation_reasoning
            }, 
            sort_keys=False
        ))

        if evaluation_score:
            return evaluation_score, evaluation_reasoning, reflexion_feedbacks
        
        joined_feedbacks = "\n###\n".join(reflexion_feedbacks)
        feedback = await self.self_reflect(
            query=task, trajectory_summary=trajectory_summary,
            evaluation_score=evaluation_score, evaluation_reasoning=evaluation_reasoning,
            previous_reflection=joined_feedbacks
        )
        print("reflexion")
        print(feedback)

        reflexion_feedbacks.append(feedback)
        return evaluation_score, evaluation_reasoning, reflexion_feedbacks
    
        
        



