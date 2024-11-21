from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.schema import HumanMessage, SystemMessage

llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",  # API version,
    azure_endpoint="https://rakbankgenaidevai.openai.azure.com/",  # Your Azure OpenAI resource URL
    api_key="db8d369a30e840b39ccdfdce4808ec7f",  # Your Azure OpenAI API key
    deployment_name="gpt-4o",
)

async def search_query(chat_history, query):
        systemMessage = f'''As an AI assistant specialized in reformulating questions based on user's prior inquiries, your objective is to generate a well-structured sentence that serves as a modified version of the provided question. The purpose of this rephrased question is to facilitate similarity search within the database, especially for subsequent inquiries.

If the given question is unrelated to the previous one, your response should mirror the exact question asked. However, when the given question is a follow-up and related to the user's chat history, your proficiency lies in crafting a rephrased question tailored to the context.

Illustrative examples to guide your task:

Example 1
--------------------------------
Previous question:
User: How to apply for a casual leave?

Given question: and the procedure?
Rephrased question: How can one request a casual leave, and what are the steps involved in the process?
---------------------------------

Example 2
--------------------------------
Previous question:
User: What will be the total working hours from Monday to Friday?

Given question: I'm not sure, can you clarify?
Rephrased question: Could you please provide further details regarding the total working hours from Monday through Friday?
---------------------------------

Example 3
--------------------------------
Previous question:
User: What is the policy for attendance and punctuality?
                  
Given question: Explain further.
Rephrased question: Could you elaborate on the guidelines regarding attendance and punctuality??
---------------------------------

Constraints:
- If the given question is related to the previous question but is itself a stand-alone or a proper question, return it as is for the rephrased question.
- If the given question and the previous question have common elements but the given question is a stand-alone or a proper question, return it as is for the rephrased question.
    '''
        userMessage=f'''Previous question:
    {chat_history}

    Given question: {query}
    Rephrased question:
    '''
    
        
        message = [
            SystemMessage(content=systemMessage),
            HumanMessage(content=userMessage)
        ]
        print(message)
        try:
            from pydantic import BaseModel, Field
            class Rephrase_Response(BaseModel):
                '''Response to show user.'''

                rephrased_query: str = Field(description="Query after rephrasing")
                query: str = Field(description="Original query given by user.")
                content: str = Field(description="Query after rephrasing")

            structured_llm = llm.with_structured_output(Rephrase_Response)


            with get_openai_callback() as cb:
                response = structured_llm.invoke(message)
            total_tokens=cb.total_tokens
            completion_tokens=cb.completion_tokens
            prompt_tokens=cb.prompt_tokens
            print(total_tokens,completion_tokens,prompt_tokens)
            print(response.content)
            return response.content
        except Exception as e:
            res=f'OpenAI Failed\nError:{e}'
        return res.strip()
    