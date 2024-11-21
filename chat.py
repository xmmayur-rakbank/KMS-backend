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
from scipy.spatial.distance import cosine
import re

from rephrasing import search_query
sessionId = "session123"
previousContext={}
previousSources={}
from pydantic import BaseModel, Field

class KMS_Response(BaseModel):
    '''Response to show user.'''
    content: str = Field(description="Actual content or response generated for question")
    answered: bool = Field(description="True if we received the answer and its present in context. False otherwise")
    is_followup: bool = Field(description="True if new query is followup of previous query False otherwise")
         

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
    openai_api_version="2023-05-15"  # API version you're using
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


    intent = ""
    engg_prompt="""
        You are helpful assistant and provides answer to the questions only based on the context provided.
        If the user query is ambiguous or can fit into multiple products , present top 3 similar products and ask for followup question for clarification like which one did you meant?
        If context is empty or irrelevant then you simply respond as "Hey there I can help you answer questions related to Rakbank internal documents, please try asking something related to it in specific manner."
        You only answer in English.
        You convert context provided in English if it is in language other than English.
        You use translated context to generate your answer.
        Important points while generating and answer:
            1. Please provide the following response in markdown format, using headers, lists and any necessary formatting to make the information easy to read. Include bullets for exceptions and conditions where applicable. Include tabs and indentations wherever required. 
            2. You generate summarized answer with heading "Summarized Answer" at the very top with word count less than 80.
            3. summarized answer is follwed by detailed answer strictly only if required with heading "Detailed Answer" which can have more heading and details but strictly relevant to the Question. 
            4. You strictly do not provide any irrelavant details, headings, points in "Detailed Answer".
            5. You generate stepwise answer wherever possible.
            6. You generate subpoints of points wherever required.
            7. Dont add statements like "In the given context....","Provided context..." etc
            8. You Answer in not more than 100 words.
            9. You strictly do not answer anything more than asked in query.
        
        Answer the question with given context.
        Question:```{Question}```
        Context: ```{Context}```
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
        documents=[]
        for x in docs:
            if(x[1] >= 0.70):
                documents.append(x[0])
        doc_strings=[format_document(doc,document_prompt) for doc in documents]
        return document_separator.join(doc_strings)
    
    # documents= retriever.get_relevant_documents(query)
    # documents = chromadbinstance.similarity_search_with_score(
    #     query,
    #     k=5,
    #     filter=filter,
    # )
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db_v1",  # Where to save data locally, remove if not necessary
        )
        
        # persist_directory="./chroma_langchain_db",
    
    # if(is_followup):
    #     context = f"""
    #     "Previous conversation" : {chat_history_for_search},
    #     "Retrieved documents" : {context}
    #     """
        

    def calculate_similarity_openai(query, context):
        query_embedding = embeddings.embed_query(query)
        context_embedding = embeddings.embed_query(context)
        # Cosine similarity calculation
        similarity = 1 - cosine(query_embedding, context_embedding)
        return similarity
    
    # Path to the JSON file where conversations are stored
    json_file_path = "conversation.json"

    # Function to load conversation history from JSON
    def load_conversation_history():
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                return json.load(file)
        else:
            return {}

    # Function to save conversation history to JSON
    def save_conversation_history(conversation_history):
        with open(json_file_path, 'w') as file:
            json.dump(conversation_history, file, indent=4)

    # Function to store a conversation for a specific sessionId
    def store_conversation(sessionId, user_query, bot_response):
        # Load the current conversation history
        conversation_history = load_conversation_history()

        # Ensure there is a list for the given sessionId
        if sessionId not in conversation_history:
            conversation_history[sessionId] = []

        # Append the new query-response pair to the session's conversation history
        conversation_history[sessionId].append({
            "user_query": user_query,
            "bot_response": bot_response.content
        })

        # Save the updated conversation history back to the JSON file
        save_conversation_history(conversation_history)

    # Function to retrieve conversation history for a specific sessionId
    def get_session_history_chat(sessionId):
        conversation_history = load_conversation_history()
        return conversation_history.get(sessionId, [])

    session_history = get_session_history_chat(sessionId)
    # print(session_history)
    # string_context = "" 
    # for msg in session_history:
    #     string_context += f'{msg["user_query"]}'

    # if(len(session_history)>0):
    #     string_context = session_history[-1]["user_query"]

    # score =calculate_similarity_openai(query , string_context)
    
    # sources=[]
    # print("score:::",score)
    conversations=[]
    conversation_text=''
    conversations = session_history[-2:]
    # query = conversations[-1]['user']
    conversations.append({'user_query':query})
    conversation_text = "\n".join([
            f"User: {msg['user_query']}\nAssistant: {msg.get('bot_response', '')}"
            for msg in conversations
        ])
        
    prompt = f"""Given the following conversation, generate a rephrased search query that captures the main information need:

    Conversation:
    {conversation_text}

    Example rephrased search queries: 
    1.	What are the benefits of the HighFlyer credit card?
    2.	What is the eligibility for a Gold Account?
    3.	What are the key features of the RAKBooster Account?

    Generate a detailed meaningful search query that best represents the user's current information need. Focus on the last user message to check if it is conversational or not. Do not change the query if it is not conversational in nature. Provide your final search query inside content.
    Current user query can be categorized as followup query if it can be answered from previous answer or its context."""

    structured_llm = llm.with_structured_output(KMS_Response) 
    from langchain.schema import HumanMessage, SystemMessage
    message = [
            HumanMessage(content=prompt)
        ] 
    is_followup=False
    with get_openai_callback() as cb:
            response = structured_llm.invoke(message)
    is_followup=response.is_followup
    query  = response.content
    if(is_followup and len(session_history) > 0):
        engg_prompt="""
        You are helpful assistant and provides answer to the followup questions only based on the context provided.
        If the user query is ambiguous or can fit into multiple products , present top 3 similar products and ask for followup question for clarification like which one did you meant?
        You provide context of previous answer to make current answer more understandable and clear.
        If context is empty or irrelevant then you simply respond as "Hey there I can help you answer questions related to Rakbank internal documents, please try asking something related to it in specific manner."
        You only answer in English.
        You convert context provided in English if it is in language other than English.
        You use translated context to generate your answer.
        Important points while generating and answer:
            1. Please provide the following response in markdown format, using headers, lists and any necessary formatting to make the information easy to read. Include bullets for exceptions and conditions where applicable. Include tabs and indentations wherever required. 
            2. You generate summarized answer with heading "Summarized Answer" at the very top with word count less than 80.
            3. summarized answer is follwed by detailed answer strictly only if required with heading "Detailed Answer" which can have more heading and details but strictly relevant to the Question. 
            4. You strictly do not provide any irrelavant details, headings, points in "Detailed Answer".
            5. You generate stepwise answer wherever possible.
            6. You generate subpoints of points wherever required.
            7. Dont add statements like "In the given context....","Provided context..." etc
            8. You Answer in not more than 100 words.
            9. You strictly do not answer anything more than asked in query.
        
        Answer the question with given context.
        """
        prompt_new = ChatPromptTemplate.from_messages([
            (
                "system",
                engg_prompt + '\n' +"{context}"
            ),
            MessagesPlaceholder(variable_name='history'),
            ("human","{input}")
        ])



        structured_llm = llm.with_structured_output(KMS_Response)
        runnable= prompt_new | structured_llm
        # RunnableWithMessageHistory was used in langchain v0.2
        # As of the v0.3 release of LangChain, we recommend that LangChain users take advantage of LangGraph persistence to incorporate memory into new LangChain applications.
        with_message_history=RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        
        context = previousContext["previousContext"]
        sources =previousSources["sources"]
        intent=identify_intent(query)

        if(intent["category"]=="Product-Specific"):
            Rule="""
                You follow below explicit rules while generating response format.
                Format Rule : 
                    •	Specific (not more than 350 words) 
                    •	If the response is longer than 100 words, then put it in a mark-down format 
                    •	Answer should be relevant to the subject-matter identified in the question, and should not include additional information related to the product that is not mentioned in the question. 
                    •	Numbers and taxes (VAT) should be included for product wherever necessary.

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
                        -   The structure of the response should be similar to a step-by-step guide. It should follow a chronological order based on how someone reading would do it.  
                """
                engg_prompt+=Rule
            
        with get_openai_callback() as cb:
            response = with_message_history.invoke(
                {
                    "context": context,
                    "input": query
                },
                config={"configurable":{"session_id":persist_directory + username, "verbose": True}}
            )

        if(not response.answered):
            engg_prompt="""
        You are helpful assistant and provides answer to the questions only based on the context provided.
        If the user query is ambiguous or can fit into multiple products , present top 3 similar products and ask for followup question for clarification like which one did you meant?
        If context is empty or irrelevant then you simply respond as "Hey there I can help you answer questions related to Rakbank internal documents, please try asking something related to it in specific manner."
        You only answer in English.
        You convert context provided in English if it is in language other than English.
        You use translated context to generate your answer.
        Important points while generating and answer:
            1. Please provide the following response in markdown format, using headers, lists and any necessary formatting to make the information easy to read. Include bullets for exceptions and conditions where applicable. Include tabs and indentations wherever required. 
            2. You generate summarized answer with heading "Summarized Answer" at the very top with word count less than 80.
            3. summarized answer is follwed by detailed answer strictly only if required with heading "Detailed Answer" which can have more heading and details but strictly relevant to the Question. 
            4. You strictly do not provide any irrelavant details, headings, points in "Detailed Answer".
            5. You generate stepwise answer wherever possible.
            6. You generate subpoints of points wherever required.
            7. Dont add statements like "In the given context....","Provided context..." etc
            8. You Answer in not more than 100 words.
            9. You strictly do not answer anything more than asked in query.
        
        Answer the question with given context.
        Question:```{Question}```
        Context: ```{Context}```
        """
            
            get_session_history(persist_directory + username)
    
            session_history = get_session_history_chat(sessionId)
            print(session_history)
            history = ""
            for msg in session_history:
                history += f'User: {msg["user_query"]}; '
            
            # rephrased_query = await search_query(history,query)
            # query = rephrased_query

            intent=identify_intent(query)

            if(intent["category"]=="Product-Specific"):
                Rule="""
                    You follow below explicit rules while generating response format.
                    Format Rule : 
                        •	Specific (not more than 350 words) 
                        •	If the response is longer than 100 words, then put it in a mark-down format 
                        •	Answer should be relevant to the subject-matter identified in the question, and should not include additional information related to the product that is not mentioned in the question. 
                        •	Numbers and taxes (VAT) should be included for product wherever necessary.

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
                            -   The structure of the response should be similar to a step-by-step guide. It should follow a chronological order based on how someone reading would do it.  
                    """
                    engg_prompt+=Rule
                
                
            documents = await vector_store.asimilarity_search_with_relevance_scores(
                query, k=7, filter=filter
            )
            sources = []
            for doc in documents:
                # print(doc.page_content)
                data=doc[0].metadata["source"]
                # print(Mapping["KFS-Rakbank.pdf"])
                print(doc[0].page_content)
                print(doc[0].metadata["page"])
                source=data
                if(doc[1]>=0.70):
                    sources.append(source)
                    sources.append(doc[0].metadata["page"])
            
            context=_combine_documents(documents)
            previousContext["previousContext"]=context
            previousSources["sources"]=sources
            
            template=ChatPromptTemplate.from_template(template=engg_prompt)
            print(template.messages[0].prompt)
            print(template.messages[0].prompt.input_variables)
            message=template.format_messages(Question=query,Context=context)
            # Default_Prompt = PromptTemplate.from_template(template="{page_content}")
            # runnable = message | llm
            with get_openai_callback() as cb:
                response = llm(message)
    else:
        get_session_history(persist_directory + username)

        documents = await vector_store.asimilarity_search_with_relevance_scores(
            query, k=7, filter=filter
        )
        sources=[]
        for doc in documents:
            # print(doc.page_content)
            data=doc[0].metadata["source"]
            # print(Mapping["KFS-Rakbank.pdf"])
            print(doc[0].page_content)
            print(doc[0].metadata["page"])
            source=data
            if(doc[1]>=0.70):
                sources.append(source)
                sources.append(doc[0].metadata["page"])

        context=_combine_documents(documents)
        previousContext["previousContext"]=context
        previousSources["sources"]=sources
        
        intent=identify_intent(query)

        if(intent["category"]=="Product-Specific"):
            Rule="""
                You follow below explicit rules while generating response format.
                Format Rule : 
                    •	Specific (not more than 350 words) 
                    •	If the response is longer than 100 words, then put it in a mark-down format 
                    •	Answer should be relevant to the subject-matter identified in the question, and should not include additional information related to the product that is not mentioned in the question. 
                    •	Numbers and taxes (VAT) should be included for product wherever necessary.

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
                        -   The structure of the response should be similar to a step-by-step guide. It should follow a chronological order based on how someone reading would do it.  
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

    store_conversation(sessionId, query, answer)
    total_tokens=cb.total_tokens
    completion_tokens=cb.completion_tokens
    prompt_tokens=cb.prompt_tokens
    print(total_tokens,completion_tokens,prompt_tokens)
    # print(answer,total_tokens,completion_tokens, prompt_tokens)
    print(answer,sources)
    # ans=answer['content']
    # intent + "\n\n" +
    return {"answer": answer, "sources":sources, "intent":intent}

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
        Also queries may contain some words from list: [eligibility, benefits, documents, conditions, interest rate, cashback, turnaround time, skywards miles, merchants, World card, Skyward card, what-is, fees]
    2. Customer-Queries:
        This category contains inputs that will elicit a step-by-step process outline that a contact center agent will have to guide a customer through (i.e. the actions will be taken by the customer on the phone, and the contact center agent will be guiding the customer on their own device using the instructions displayed on the screen). This response will have two layers of information – what the contact center agent should tell the customer and what the customer should be seeing on their end when they perform the actions that the contact center agent tells them. 
        Also queries may contain some words from list: [activate, set up, reset, process, new card, statement, liability certificate, renewal, block, digital pathway, PIN, app, balance, claim, how-to]
    3. Customer-Requests:
        This category contains inputs that will elicit a step-by-step process outline that the contact center agent has to undertake by themselves to conduct an action on behalf of the customer. This will include steps that need to be undertaken on their own systems, like Finacle & IBPS, and will be descriptive enough to help them navigate those systems to fulfill the customer’s request. 
        Also queries may contain some words from list:[reversal, process, closure, manual, dispute, credit limit, IBPS, Finacle, service request, cancellation, overlimit fee, late payment fee, reverse charge, how-to]
    4. General:
        This category is choosen if it does not falls into above categories.

    output format:{{"category":<Product-Specific/Customer-Queries/Customer-Requests>}}
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