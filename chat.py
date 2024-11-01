from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.callbacks import get_openai_callback
from langchain.schema import format_document
from OnedriveMapping import Mapping
import os
import glob
import json

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

# embeddings = AzureOpenAIEmbeddings(
#     deployment="text-embedding-ada-002",  # Azure OpenAI deployment name
#     model="text-embedding-ada-002",  # Model you want to use
#     api_key="643406f5cf994e82bf1dadecf0e120f1",  # Your Azure OpenAI API key
#     azure_endpoint="https://gpt-demo-openai.openai.azure.com/",  # Your Azure OpenAI resource URL
#     openai_api_version="2023-12-01-preview"  # API version you're using
# )

# llm = AzureChatOpenAI(
#     openai_api_version="2023-12-01-preview",  # API version,
#     azure_endpoint="https://gpt-demo-openai.openai.azure.com/",  # Your Azure OpenAI resource URL
#     api_key="643406f5cf994e82bf1dadecf0e120f1",  # Your Azure OpenAI API key
#     deployment_name="gpt-4",
# )

embeddings = AzureOpenAIEmbeddings(
    deployment="text-data-002",  # Azure OpenAI deployment name
    model="text-embedding-ada-002",  # Model you want to use
    api_key="db8d369a30e840b39ccdfdce4808ec7f",  # Your Azure OpenAI API key
    azure_endpoint="https://rakbankgenaidevai.openai.azure.com/",  # Your Azure OpenAI resource URL
    openai_api_version="2023-12-01-preview"  # API version you're using
)

llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",  # API version,
    azure_endpoint="https://rakbankgenaidevai.openai.azure.com/",  # Your Azure OpenAI resource URL
    api_key="db8d369a30e840b39ccdfdce4808ec7f",  # Your Azure OpenAI API key
    deployment_name="gpt-4o",
)

# llm = AzureOpenAI(
#     openai_api_version="2023-03-15-preview",  # API version,
#     azure_endpoint="https://enterprisesearch-openai-demo.openai.azure.com/",  # Your Azure OpenAI resource URL
#     api_key="3cc7437b3389437e9c0668bd8c3f3ab5",  # Your Azure OpenAI API key
#     deployment_name="testdeployment001",
# )

async def chatting(query,department , history, is_followup, username):
    # chat_history_for_search = get_chat_history_as_text(history.ChatHistory)
    
    persist_directory="chroma_db"
    chromadbinstance = Chroma(
         embedding_function=embeddings, persist_directory=persist_directory
    )
    list =[]
    for dept in department:
        list.append({'department':dept})

    filter = {}
    if(len(list)>1):
        filter = {
            '$or': list
        }
    else:
        filter = list[0]

    retriever= chromadbinstance.as_retriever(search_kwargs={'k': 10,  'filter': filter })

    engg_prompt="""
    You are helpful assiatsnt and provides answer to the questions only based on the context provided.
    If context is emplty then simply respond as "Provided context dont have answer to this question."
    You only answer in english.
    You convert context provided in English if it is in language other than English.
    You use translated context to generate your answer.
    Important points while generating and answer:
        1. Please provide the following response in markdown format, using headers, lists, and any necessary formatting to make the information easy to read. Include bullets for exceptions and conditions where applicable.
        2. You generate stepwise answer wherever possible.
        3. Your steps might contain points like 'Eligibility and Requirements', 'Steps to open account' etc.
        4. You generate subpoints of points wherever required.
        5. You generate answers in points and subpoints only if necessary.
    """

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id == persist_directory + username:
            if session_id not in store:
                store[session_id]=ChatMessageHistory()
                print(f"stored chat message history {ChatMessageHistory()}")
                return store[session_id]
            
            history = store[session_id]

            limited_history = history.messages[-6:]  # Adjust the slice as needed

            store[session_id].messages=limited_history
            print(store[session_id])
            return store[session_id]
        else:
            return ChatMessageHistory()
        
    Default_Prompt = PromptTemplate.from_template(template="{page_content}")

    def _combine_documents(docs, document_prompt=Default_Prompt, document_separator="\n\n"):
        doc_strings=[format_document(doc,document_prompt) for doc in docs]
        return document_separator.join(doc_strings)
    
    prompt_new = ChatPromptTemplate.from_messages([
            (
                "system",
                engg_prompt + '\n' +"{context}"
            ),
            MessagesPlaceholder(variable_name='history'),
            ("human","{input}")
    ])


    runnable= prompt_new | llm
    # RunnableWithMessageHistory was used in langchain v0.2
    # As of the v0.3 release of LangChain, we recommend that LangChain users take advantage of LangGraph persistence to incorporate memory into new LangChain applications.
    with_message_history=RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    documents= retriever.get_relevant_documents(query)
    sources=[]
    for doc in documents:
        # print(doc.page_content)
        data=doc.metadata["source"]
        print(Mapping["KFS-Rakbank.pdf"])
        source=data
        sources.append(source)
        sources.append(doc.metadata["page"])
    # print(documents)
    context=_combine_documents(documents)
    # if(is_followup):
    #     context = f"""
    #     "Previous conversation" : {chat_history_for_search},
    #     "Retrieved documents" : {context}
    #     """
        
    # print(context)
    if(is_followup):
        
        with get_openai_callback() as cb:
            response = with_message_history.invoke(
                {
                    "context": context,
                    "input": query
                },
                config={"configurable":{"session_id":persist_directory + username, "verbose": True}}
            )
    else:
        get_session_history(persist_directory + username)
    #     engg_prompt="""
    # You are helpful assiatsnt and provides answer to the questions only based on the context provided.
    # If context is emplty then simply respond as "Provided context dont have answer to this question."
    # You only answer in english.
    # You convert context provided in English if it is in language other than English.
    # You use translated context to generate your answer.
    # Important points while generating and answer:
    #     1. Please provide the following response in markdown format, using headers, lists, and any necessary formatting to make the information easy to read. Include bullets for exceptions and conditions where applicable.
    #     2. You generate stepwise answer wherever possible.
    #     3. Your steps might contain points like 'Eligibility and Requirements', 'Steps to open account' etc.
    #     4. You generate subpoints of points wherever required.
    #     5. You generate answers in points and subpoints only if necessary.
    
    # Answer the question with given context.
    # Question:```{Question}```
    # Context: ```{Context}```
    # """
        engg_prompt="""
    You are helpful assistant and provides answer to the questions only based on the context provided.
    If context is empty or irrelevant then you simply respond as "Hey there I can help you answer questions related to Rakbank internal documents, please try asking something related to it."
    You only answer in English.
    You convert context provided in English if it is in language other than English.
    You use translated context to generate your answer.
    Important points while generating and answer:
        1. Please provide the following response in markdown format, using headers, lists and any necessary formatting to make the information easy to read. Include bullets for exceptions and conditions where applicable.
	    2. You generate summarized answer with heading "Summarized Answer" at the very top with word count less than 80.
        3. summarized answer is follwed by detailed answer with heading "Detailed Answer" which can have more heading and details but strictly relevant to the Question. There is no word count limit on "Detailed Answer". Show Scripts and steps as it is from context without any alteration.
        4. You always add subheading and steps in Detailed answer whenever possible.
        5. You generate stepwise answer wherever possible.
        6. Your steps might contain points like 'Eligibility and Requirements', 'Steps to open account' etc.
        7. You generate subpoints of points wherever required.
        8. You generate answers in points and subpoints only if necessary.
        9. Dont add statements like "In the given context....","Provided context..." etc
    
    Answer the question with given context.
    Question:```{Question}```
    Context: ```{Context}```
    """
        intent=identify_intent(query)

        if(intent["category"]=="Product-Specific"):
            Rule="""
                You follow below explicit rules while generating response format.
                Format Rule : 
                    -	Specific (limited to 350 words to avoid it being too verbose)
                    -	General (no customer-specific or situation-specific information)
                    -	If the response is longer than 100 words, then put it in a mark-down format
            """
            engg_prompt+=Rule
        elif(intent["category"]=="Customer-Queries"):
            Rule="""
                You follow below explicit rules while generating response format.
                Format Rule : 
                    -	Information should be step-by-step, displayed as a process
                    -	Response should be displayed in a conversational tone and the steps should be simple to understand
                    -	Response should include the expected outcome (what the customer should see when they follow the steps that the contact center agent gives them
            """
            engg_prompt+=Rule
        elif(intent["category"]=="Customer-Requests"):
            Rule="""
                You follow below explicit rules while generating response format.
                Format Rule : 
                    -	Information should be step-by-step
                    -	Should include information about how to navigate internal systems (more technical jargon – will be enhanced if we also upload a glossary) 
            """
            engg_prompt+=Rule
        
        template=ChatPromptTemplate.from_template(template=engg_prompt)
        print(template.messages[0].prompt)
        print(template.messages[0].prompt.input_variables)
        message=template.format_messages(Question=query,Context=context)
        # Default_Prompt = PromptTemplate.from_template(template="{page_content}")
        # runnable = message | llm
        with get_openai_callback() as cb:
            response = llm(message)
    answer= response
    total_tokens=cb.total_tokens
    completion_tokens=cb.completion_tokens
    prompt_tokens=cb.prompt_tokens
    print(total_tokens,completion_tokens,prompt_tokens)
    # print(answer,total_tokens,completion_tokens, prompt_tokens)
    print(answer,sources)
    return {"answer":answer, "sources":sources}

def get_chat_history_as_text(history):
    history_text=""
    for history_item in reversed(history):
        history_text=(
            """<|im_start|>user"""+"\n"+"user"+": "+history_item.user+"\n"+"""<|im_end|>"""+"\n"
            """<|im_start|>assistant"""+"\n"+(history_item.assistant+"""<|im_end|>""" if history_item.assistant else "")+"\n"+
            history_text
        )
        if(len(history_text))>1000:
            break
    return history_text

def identify_intent(query):
    propmt="""
    You are a category classifier assistant who help classify the query into given categories.
    STRICTLY return output in JSON format.
    Categories must be from [ 'Product-Specific', 'Customer-Queries', 'Customer-Requests', 'General']
    You strictly use below rules and conditions to identify the category:
    1. Product-Specific:
        This category contains inputs that are requesting static information about a product. 
        Also queries may contain some words from list: [eligibility, benefits, documents, conditions, interest rate, cashback, turnaround time, skywards miles, merchants, World card, Skyward card, what-is]
    2. Customer-Queries:
        This category contains inputs that will elicit a step-by-step process outline that a contact center agent will have to guide a customer through (i.e. the actions will be taken by the customer on the phone, and the contact center agent will be guiding the customer on their own device using the instructions displayed on the screen). This response will have two layers of information – what the contact center agent should tell the customer and what the customer should be seeing on their end when they perform the actions that the contact center agent tells them. 
        Also queries may contain some words from list: [activate, set up, reset, process, new card, statement, liability certificate, renewal, block, digital pathway, PIN, app, balance, claim, how-to]
    3. Customer-Requests:
        This category contains inputs that will elicit a step-by-step process outline that the contact center agent has to undertake by themselves to conduct an action on behalf of the customer. This will include steps that need to be undertaken on their own systems, like Finacle & IBPS, and will be descriptive enough to help them navigate those systems to fulfill the customer’s request. 
        Also queries may contain some words from list:[reversal, process, closure, manual, dispute, credit limit, IBPS, Finacle, service request, cancellation, overlimit fee, late payment fee, reverse charge, how-to]
    4. General:
        This category is choosen if it does not falls into above categories.

    output format:{{"category":<Product-Specific/Customer-Queries/Customer-Requests/COMPARE/GAPANALYSIS/GREETING>}}
    Question:```{query}```
    """
    template=ChatPromptTemplate.from_template(template=propmt)
    
    message=template.format_messages(query=query)
    with get_openai_callback() as cb:
            response = llm(message)
    import json
    data=response.content.replace('json\n','').replace('\n','').replace('```','')
    data1=json.loads(data)
    
    return data1