from pydantic_settings import BaseSettings
from pydantic import Field

class Credentials(BaseSettings):
    OPENAI_API_KEY: str = Field(..., env='OPENAI_API_KEY')


