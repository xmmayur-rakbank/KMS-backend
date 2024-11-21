from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",  # API version,
    azure_endpoint="https://rakbankgenaidevai.openai.azure.com/",  # Your Azure OpenAI resource URL
    api_key="db8d369a30e840b39ccdfdce4808ec7f",  # Your Azure OpenAI API key
    deployment_name="gpt-4o",
)

import json
from textwrap import dedent
from openai import OpenAI
# client = OpenAI()

# MODEL = "gpt-4o-2024-08-06"

from typing import Optional

from pydantic import BaseModel, Field


class Joke(BaseModel):
    '''Joke to tell user.'''

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


structured_llm = llm.with_structured_output(Joke)
response = structured_llm.invoke("Tell me a joke about cats")
print(response["setup"])

            # Joke(
            #     setup="Why was the cat sitting on the computer?",
            #     punchline="To keep an eye on the mouse!",
            #     rating=None,
            # )