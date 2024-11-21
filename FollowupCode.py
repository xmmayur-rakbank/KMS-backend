
import torch
import polars as pl
import re
import json
import async_timeout
import asyncio
from typing import List, Dict
import pickle
from contextlib import asynccontextmanager
import numpy as np

from fuzzywuzzy import process

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from openai import AsyncAzureOpenAI, AzureOpenAI

from constants_secrets import OPENAI_KEY
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



CONFIDENCE_THRESHOLD = 0.4
GENERATION_TIMEOUT_SEC = 60


models_and_data = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # with open("embeddings/embeddings.pkl", "rb") as f:
    # with open("embeddings/only_eng_embeddings.pkl", "rb") as f:
    # data = pickle.load(f)
    # models_and_data['embeddings'] = data['embeddings']
    # models_and_data['df'] = data['df']
    # sentences = models_and_data['df']['Text'].to_list()
    # models_and_data['sentences'] = sentences
    # print(sentences[:5])
    # print(data['df'])

    with open("embeddings/all_MiniLM_L6_v2_supporting_links_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        models_and_data['all_MiniLM_L6_v2_supporting_links_embeddings'] = data['embeddings']
        models_and_data['df'] = data['df']
        sentences = models_and_data['df']['Service Type'].to_list()
        models_and_data['sentences'] = sentences



    # with open("embeddings/bge_large_en_v1.5_only_eng_embeddings.pkl", "rb") as f:
    # data = pickle.load(f)
    # models_and_data['bge_large_en_v1.5_only_eng_embeddings'] = data['embeddings']

    with open("embeddings/bge_large_en_v1.5_supporting_links_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        models_and_data['bge_large_en_v1.5_supporting_links_embeddings'] = data['embeddings']

    # with open("embeddings/all_mpnet_v2_only_eng_embeddings.pkl", "rb") as f:
    # data = pickle.load(f)
    # models_and_data['all_mpnet_v2_only_eng_embeddings'] = data['embeddings']

    with open("embeddings/mpnet_base_v2_supporting_links_embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        models_and_data['mpnet_base_v2_supporting_links_embeddings'] = data['embeddings']

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_embeddings = tfidf_vectorizer.fit_transform(sentences)

    models_and_data['tfidf_vectorizer'] = tfidf_vectorizer
    models_and_data['tfidf_embeddings'] = tfidf_embeddings
    models_and_data["all-minilm-l6-v2"] = SentenceTransformer("all-MiniLM-L6-v2")
    models_and_data["all-mpnet-base-v2"] = SentenceTransformer("all-mpnet-base-v2")
    models_and_data["BAAI/bge-large-en-v1.5"] = SentenceTransformer("BAAI/bge-large-en-v1.5")
    # models_and_data["openai_client"] = OpenAI(api_key=OPENAI_KEY)
    # models_and_data["async_openai_client"] = AsyncOpenAI(api_key=OPENAI_KEY)
    models_and_data["openai_client"] = AzureOpenAI(azure_endpoint=endpoint, api_version=api_version, api_key=key)
    models_and_data["async_openai_client"] = AsyncAzureOpenAI(azure_endpoint=endpoint, api_version=api_version, api_key=key)
    yield
    models_and_data.clear()


app = FastAPI(lifespan=lifespan)
# app.mount("/static", StaticFiles(directory="static"), name='static')
app.mount("/static", StaticFiles(directory="build/static"), name="static")

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:9999",
    "http://172.50.49.83",
    "http://172.50.49.83:3000",
    "http://172.50.49.83:9999",
    "http://172.50.55.245",
    "http://172.50.55.245:9999",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates("templates")

    
@app.get("/ping", response_class=JSONResponse)
async def ping():
    return JSONResponse({"message": "pong"})

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return FileResponse("build/index.html")

@app.get("/favicon.ico")
async def favicon(request: Request):
    return FileResponse("build/favicon.ico")
    return templates.TemplateResponse("home.html", {"request": request})





class Conversations(BaseModel):
    conversations: List[Dict[str, str]]
    top_k: int = 10


@app.post("/rag")
async def rag(request: Conversations):
    conversations = request.conversations
    query = conversations[-1]['user']

    if len(conversations) == 1:
        search_query = conversations[0]['user']
    else:
        conversation_text = "\n".join([
            f"User: {msg['user']}\nAssistant: {msg.get('assistant', '')}"
            for msg in conversations
        ])
        
        prompt = f"""Given the following conversation, generate a focused search query that captures the main information need:

    Conversation:
    {conversation_text}

    Example: 
    Transfer funds - Within Own Accounts
    Transfer funds - Within RAKBank
    Transfer funds - Within UAE
    Cheque Book Request
    RAKMoney Transfer - Account remittances to India
    Remittance cancellation
    Obtain Swift Copy
    Apply for Debit Card


    Generate a concise search query that best represents the user's current information need. Focus on the last user message to check if it is conversational or not. Do not change the query if it is not conversational in nature. Start with thinking steps inside <thinking></thinking> tag in step by step way before formulating you final search query. Provide your final search query inside <search_query></search_query> tag."""
        


        response = models_and_data["openai_client"].chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        
        llm_search_query = response.choices[0].message.content.strip()
        print(llm_search_query)
        search_query = re.findall(r"<search_query>(.*?)</search_query>", llm_search_query, re.DOTALL)
        if search_query:
            search_query = search_query[0]
        else:
            return JSONResponse(content={
                "answer": "I'm sorry, but there is no relevant information in the provided context to answer your question."
            })

    with torch.no_grad():
        q_em = models_and_data["all-minilm-l6-v2"].encode(search_query.lower())
    semantic_search_results = semantic_search(q_em, models_and_data['all_MiniLM_L6_v2_supporting_links_embeddings'], top_k=request.top_k)[0]
    print(semantic_search_results)


    with torch.no_grad():
        q_em = models_and_data["all-mpnet-base-v2"].encode(search_query.lower())
    all_mpnet_v2_only_eng_embeddings_semantic_search_results = semantic_search(q_em, models_and_data['mpnet_base_v2_supporting_links_embeddings'], top_k=request.top_k)[0]
    print(all_mpnet_v2_only_eng_embeddings_semantic_search_results)

    with torch.no_grad():
        q_em = models_and_data["BAAI/bge-large-en-v1.5"].encode(search_query.lower())
    bge_large_en_v1_5_only_eng_embeddings_only_eng_embeddings_semantic_search_results = semantic_search(q_em, models_and_data['bge_large_en_v1.5_supporting_links_embeddings'], top_k=request.top_k)[0]
    print(bge_large_en_v1_5_only_eng_embeddings_only_eng_embeddings_semantic_search_results)




    tfidf_q_em = models_and_data['tfidf_vectorizer'].transform([search_query.lower()])
    tfidf_search_results = cosine_similarity(models_and_data['tfidf_embeddings'], tfidf_q_em)
    tfidf_search_results = tfidf_search_results.flatten()
    top_n_indices = np.argsort(tfidf_search_results)[-request.top_k:][::-1]

    context = []

    fuzzy_wuzzy_search_results = process.extract(search_query, models_and_data['sentences'], limit=5)

    for res in fuzzy_wuzzy_search_results:
        nearest_sentence = res[0]
        web_login_path = models_and_data['df'].filter(pl.col('Service Type') == nearest_sentence)['Web Login Path'][0]
        mobile_login_path = models_and_data['df'].filter(pl.col('Service Type') == nearest_sentence)['Mobile Login Path'][0]
        # nearest_intent = models_and_data['df'].filter(pl.col('Service Type') == nearest_sentence)['Actual Intent'][0]
        score = res[1]
        if score > 80:
            context.append({
                "search_type": "WRatio",
                "model": "fuzzywuzzy",
                "nearest_sentence": nearest_sentence,
                "web_login_path": web_login_path,
                "mobile_login_path": mobile_login_path,
                # "nearest_intent": nearest_intent,
                "score": score / 100.0,
            })



    for idx in top_n_indices:
        nearest_sentence = models_and_data['df'].row(idx)[0]
        web_login_path = models_and_data['df'].row(idx)[1]
        mobile_login_path = models_and_data['df'].row(idx)[2]
        score = float(tfidf_search_results[idx])
        if score > CONFIDENCE_THRESHOLD:
            context.append({
                "search_type": "cosine_similarity",
                "model": "tfidf",
                "nearest_sentence": nearest_sentence,
                "web_login_path": web_login_path,
                "mobile_login_path": mobile_login_path,
                # "nearest_intent": nearest_intent,
                "score": score,
            })
        print(f"Document index: {idx}, Score: {score}")
    


    print(semantic_search_results)
    print(tfidf_q_em)
    print(tfidf_search_results)


    
    for res in semantic_search_results:
        corpus_id = res['corpus_id']
        score = res['score']
        nearest_sentence = models_and_data['df'].row(corpus_id)[0]
        web_login_path = models_and_data['df'].row(corpus_id)[1]
        mobile_login_path = models_and_data['df'].row(corpus_id)[2]
        # nearest_intent = models_and_data['df'].row(corpus_id)[1]
        if score > CONFIDENCE_THRESHOLD:
            context.append({
                "search_type": "cosine_similarity",
                "model": "sentence_transformers",
                "nearest_sentence": nearest_sentence,
                "web_login_path": web_login_path,
                "mobile_login_path": mobile_login_path,
                # "nearest_intent": nearest_intent,
                "score": score,
            })

    for res in all_mpnet_v2_only_eng_embeddings_semantic_search_results:
        corpus_id = res['corpus_id']
        score = res['score']
        nearest_sentence = models_and_data['df'].row(corpus_id)[0]
        web_login_path = models_and_data['df'].row(corpus_id)[1]
        mobile_login_path = models_and_data['df'].row(corpus_id)[2]
        # nearest_intent = models_and_data['df'].row(corpus_id)[1]
        if score > CONFIDENCE_THRESHOLD:
            context.append({
                "search_type": "cosine_similarity",
                "model": "sentence_transformers_all_mpnet_base_v2",
                "nearest_sentence": nearest_sentence,
                "web_login_path": web_login_path,
                "mobile_login_path": mobile_login_path,
                # "nearest_intent": nearest_intent,
                "score": score,
            })

    for res in all_mpnet_v2_only_eng_embeddings_semantic_search_results:
        corpus_id = res['corpus_id']
        score = res['score']
        nearest_sentence = models_and_data['df'].row(corpus_id)[0]
        web_login_path = models_and_data['df'].row(corpus_id)[1]
        mobile_login_path = models_and_data['df'].row(corpus_id)[2]
        # nearest_intent = models_and_data['df'].row(corpus_id)[1]
        if score > CONFIDENCE_THRESHOLD:
            context.append({
                "search_type": "cosine_similarity",
                "model": "sentence_transformers_all_mpnet_base_v2",
                "nearest_sentence": nearest_sentence,
                "web_login_path": web_login_path,
                "mobile_login_path": mobile_login_path,
                # "nearest_intent": nearest_intent,
                "score": score,
            })

    for res in bge_large_en_v1_5_only_eng_embeddings_only_eng_embeddings_semantic_search_results:
        corpus_id = res['corpus_id']
        score = res['score']
        nearest_sentence = models_and_data['df'].row(corpus_id)[0]
        web_login_path = models_and_data['df'].row(corpus_id)[1]
        mobile_login_path = models_and_data['df'].row(corpus_id)[2]
        # nearest_intent = models_and_data['df'].row(corpus_id)[1]
        if score > CONFIDENCE_THRESHOLD:
            context.append({
                "search_type": "cosine_similarity",
                "model": "sentence_transformers_bge_large_en_v1.5",
                "nearest_sentence": nearest_sentence,
                "web_login_path": web_login_path,
                "mobile_login_path": mobile_login_path,
                # "nearest_intent": nearest_intent,
                "score": score,
            })
    

    if not context:
        return JSONResponse(content={
            "query": query,
            "search_query": search_query,
            "context": context,
            "answer": "I'm sorry, but there is no relevant information in the provided context to answer your question."
        })


    context = sorted(context, key=lambda d: d['score'], reverse=True)[:10]
    print(context)

    unique_sentences = {}
    for item in context:
        sentence = item["nearest_sentence"]
        if sentence not in unique_sentences or item["score"] > unique_sentences[sentence]["score"]:
            unique_sentences[sentence] = item

    # Convert the dictionary back to a list while maintaining the original format
    unique_context = [unique_sentences[sentence] for sentence in unique_sentences]


    # Format context for the prompt
    context_text = "\n\n".join([
        f"Nearest Sentence {i+1}: {item['nearest_sentence']}\nWeb Login Path {i+1}: {item['web_login_path']}\nMobile Login Path {i+1}: {item['mobile_login_path']}"
        for i, item in enumerate(unique_context)
    ])
    
    prompt = f"""You are RAKBank Service Navigation Assistant! You are here to help customers find the correct login paths for various banking services on both web and mobile platforms. Customers can ask you about any of the following services in the provided context, and you will provide precise and correct steps to them. Start with thinking steps inside <thinking></thinking> tag in a step-by-step way on how you will formulate answer. Your final answer should be between <answer></answer> tag and in markdown only. If you cannot find the answer in the provide context or not able to understand what customer is saying ask a follow up question based on the context or if something is not available in context, just say I don't have the answer between the answer tags only. Do not provide any information other than what is specifically asked by the customer. Understand the Customer's Question and Customer's Conversational Question when formulating final answer. 

Question: {query}
DB Conversational Search Query Based on Past Conversation: {search_query}

Context:
{context_text}

Answer the customer question using only the provided context and the DB Conversational Search Query Based on Past Conversation. Be honest and concise. End with a 3 similar more personalized follow-up question based on related context and user question inside same answer tag in markdown format"""
    
    print(prompt)
    
    response = models_and_data["openai_client"].chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )



    llm_answer = response.choices[0].message.content
    print(llm_answer)
    answer = re.findall(r"<answer>(.*?)</answer>", llm_answer, re.DOTALL)
    if answer:
        answer = answer[0].strip()
    else:
        answer = llm_answer

    llm_english_answer = re.findall(r"<english_answer>(.*?)</english_answer>", llm_answer, re.DOTALL)
    if llm_english_answer:
        english_answer = llm_english_answer[0]
    else:
        english_answer = ""


    
    return JSONResponse(content={
        "query": query,
        "search_query": search_query,
        "context": context,
        "answer": answer,
        "english_answer": english_answer,
    })





async def stream_generator(response, meta_response):
    """Yield the content between <answer> and </answer> tags by handling chunked data responses."""

    # yield json.dumps(meta_response)
    # yield "\n"

    within_answer_tag = False
    accumulated_content = "" # To accumulate the whole content
    answer_content = "" # To accumulate content inside <answer> tags

    try:
        async with async_timeout.timeout(GENERATION_TIMEOUT_SEC):
            async for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content

                if content is not None:
                    accumulated_content += content

                    # Check if we have found an opening tag
                    if "<answer>" in accumulated_content and not within_answer_tag:
                        within_answer_tag = True # Start collecting answer
                        # Remove anything before the <answer> tag
                        accumulated_content = accumulated_content.split("<answer>", 1)[1]

                    # If within the <answer> tag, accumulate answer content
                    if within_answer_tag:
                        answer_content += accumulated_content
                        accumulated_content = "" # Reset for the next chunk

                    # Check if the closing </answer> tag is found
                    if within_answer_tag and "</answer>" in answer_content:
                        # Extract only content before the closing tag
                        final_answer = answer_content.split("</answer>", 1)[0]
                        yield final_answer

                        # Reset the state for next possible <answer> block
                        within_answer_tag = False
                        answer_content = "" # Clear for the next answer

    except async_timeout.Timeout:
        yield 'Timeout while waiting for the response.'


@app.post("/stream_rag")
async def stream_rag(request: Conversations):
    conversations = request.conversations
    query = conversations[-1]['user']

    if len(conversations) == 1:
        search_query = conversations[0]['user']
    else:
        conversation_text = "\n".join([
            f"User: {msg['user']}\nAssistant: {msg.get('assistant', '')}"
            for msg in conversations
        ])
        
        prompt = f"""Given the following conversation, generate a focused search query that captures the main information need:

    Conversation:
    {conversation_text}

    Example: 
    Transfer funds - Within Own Accounts
    Transfer funds - Within RAKBank
    Transfer funds - Within UAE
    Cheque Book Request
    RAKMoney Transfer - Account remittances to India
    Remittance cancellation
    Obtain Swift Copy
    Apply for Debit Card


    Generate a concise search query that best represents the user's current information need. Focus on the last user message to check if it is conversational or not. Do not change the query if it is not conversational in nature. Start with thinking steps inside <thinking></thinking> tag in step by step way before formulating you final search query. Provide your final search query inside <search_query></search_query> tag."""
        


        response = models_and_data["openai_client"].chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        
        llm_search_query = response.choices[0].message.content.strip()
        print(llm_search_query)
        search_query = re.findall(r"<search_query>(.*?)</search_query>", llm_search_query, re.DOTALL)
        if search_query:
            search_query = search_query[0]
        else:
            return JSONResponse(content={
                "answer": "I'm sorry, but there is no relevant information in the provided context to answer your question."
            })

    with torch.no_grad():
        q_em = models_and_data["all-minilm-l6-v2"].encode(search_query.lower())
    semantic_search_results = semantic_search(q_em, models_and_data['all_MiniLM_L6_v2_supporting_links_embeddings'], top_k=request.top_k)[0]
    print(semantic_search_results)


    with torch.no_grad():
        q_em = models_and_data["all-mpnet-base-v2"].encode(search_query.lower())
    all_mpnet_v2_only_eng_embeddings_semantic_search_results = semantic_search(q_em, models_and_data['mpnet_base_v2_supporting_links_embeddings'], top_k=request.top_k)[0]
    print(all_mpnet_v2_only_eng_embeddings_semantic_search_results)

    with torch.no_grad():
        q_em = models_and_data["BAAI/bge-large-en-v1.5"].encode(search_query.lower())
    bge_large_en_v1_5_only_eng_embeddings_only_eng_embeddings_semantic_search_results = semantic_search(q_em, models_and_data['bge_large_en_v1.5_supporting_links_embeddings'], top_k=request.top_k)[0]
    print(bge_large_en_v1_5_only_eng_embeddings_only_eng_embeddings_semantic_search_results)




    tfidf_q_em = models_and_data['tfidf_vectorizer'].transform([search_query.lower()])
    tfidf_search_results = cosine_similarity(models_and_data['tfidf_embeddings'], tfidf_q_em)
    tfidf_search_results = tfidf_search_results.flatten()
    top_n_indices = np.argsort(tfidf_search_results)[-request.top_k:][::-1]

    context = []

    fuzzy_wuzzy_search_results = process.extract(search_query, models_and_data['sentences'], limit=5)

    for res in fuzzy_wuzzy_search_results:
        nearest_sentence = res[0]
        # nearest_intent = models_and_data['df'].filter(pl.col('Text') == nearest_sentence)['Actual Intent'][0]
        web_login_path = models_and_data['df'].filter(pl.col('Service Type') == nearest_sentence)['Web Login Path'][0]
        mobile_login_path = models_and_data['df'].filter(pl.col('Service Type') == nearest_sentence)['Mobile Login Path'][0]
        score = res[1]
        if score > 80:
            context.append({
                "search_type": "WRatio",
                "model": "fuzzywuzzy",
                "nearest_sentence": nearest_sentence,
                "web_login_path": web_login_path,
                "mobile_login_path": mobile_login_path,
                # "nearest_intent": nearest_intent,
                "score": score / 100.0,
            })



    for idx in top_n_indices:
        nearest_sentence = models_and_data['df'].row(idx)[0]
        web_login_path = models_and_data['df'].row(idx)[1]
        mobile_login_path = models_and_data['df'].row(idx)[2]
        # nearest_intent = models_and_data['df'].row(idx)[1]
        score = float(tfidf_search_results[idx])
        if score > CONFIDENCE_THRESHOLD:
            context.append({
                "search_type": "cosine_similarity",
                "model": "tfidf",
                "nearest_sentence": nearest_sentence,
                "web_login_path": web_login_path,
                "mobile_login_path": mobile_login_path,
                # "nearest_intent": nearest_intent,
                "score": score,
            })
        print(f"Document index: {idx}, Score: {score}")
    


    print(semantic_search_results)
    print(tfidf_q_em)
    print(tfidf_search_results)


    
    for res in semantic_search_results:
        corpus_id = res['corpus_id']
        score = res['score']
        nearest_sentence = models_and_data['df'].row(corpus_id)[0]
        web_login_path = models_and_data['df'].row(corpus_id)[1]
        mobile_login_path = models_and_data['df'].row(corpus_id)[2]

        # nearest_intent = models_and_data['df'].row(corpus_id)[1]
        if score > CONFIDENCE_THRESHOLD:
            context.append({
                "search_type": "cosine_similarity",
                "model": "sentence_transformers",
                "nearest_sentence": nearest_sentence,
                "web_login_path": web_login_path,
                "mobile_login_path": mobile_login_path,
                # "nearest_intent": nearest_intent,
                "score": score,
            })

    for res in all_mpnet_v2_only_eng_embeddings_semantic_search_results:
        corpus_id = res['corpus_id']
        score = res['score']
        nearest_sentence = models_and_data['df'].row(corpus_id)[0]
        web_login_path = models_and_data['df'].row(corpus_id)[1]
        mobile_login_path = models_and_data['df'].row(corpus_id)[2]

        # nearest_intent = models_and_data['df'].row(corpus_id)[1]
        if score > CONFIDENCE_THRESHOLD:
            context.append({
                "search_type": "cosine_similarity",
                "model": "sentence_transformers_all_mpnet_base_v2",
                "nearest_sentence": nearest_sentence,
                "web_login_path": web_login_path,
                "mobile_login_path": mobile_login_path,
                # "nearest_intent": nearest_intent,
                "score": score,
            })

    for res in all_mpnet_v2_only_eng_embeddings_semantic_search_results:
        corpus_id = res['corpus_id']
        score = res['score']
        nearest_sentence = models_and_data['df'].row(corpus_id)[0]
        web_login_path = models_and_data['df'].row(corpus_id)[1]
        mobile_login_path = models_and_data['df'].row(corpus_id)[2]

        # nearest_intent = models_and_data['df'].row(corpus_id)[1]
        if score > CONFIDENCE_THRESHOLD:
            context.append({
                "search_type": "cosine_similarity",
                "model": "sentence_transformers_all_mpnet_base_v2",
                "nearest_sentence": nearest_sentence,
                "web_login_path": web_login_path,
                "mobile_login_path": mobile_login_path,
                # "nearest_intent": nearest_intent,
                "score": score,
            })

    for res in bge_large_en_v1_5_only_eng_embeddings_only_eng_embeddings_semantic_search_results:
        corpus_id = res['corpus_id']
        score = res['score']
        nearest_sentence = models_and_data['df'].row(corpus_id)[0]
        web_login_path = models_and_data['df'].row(corpus_id)[1]
        mobile_login_path = models_and_data['df'].row(corpus_id)[2]

        # nearest_intent = models_and_data['df'].row(corpus_id)[1]
        if score > CONFIDENCE_THRESHOLD:
            context.append({
                "search_type": "cosine_similarity",
                "model": "sentence_transformers_bge_large_en_v1.5",
                "nearest_sentence": nearest_sentence,
                "web_login_path": web_login_path,
                "mobile_login_path": mobile_login_path,
                # "nearest_intent": nearest_intent,
                "score": score,
            })

    if not context:
        return JSONResponse(content={
            "query": query,
            "search_query": search_query,
            "context": context,
            "answer": "I'm sorry, but there is no relevant information in the provided context to answer your question."
        })


    context = sorted(context, key=lambda d: d['score'], reverse=True)[:request.top_k]
    print(context)


    # Format context for the prompt
    context_text = "\n\n".join([
        f"Nearest Sentence {i+1}: {item['nearest_sentence']}\nWeb Login Path {i+1}: {item['web_login_path']}\nMobile Login Path {i+1}: {item['mobile_login_path']}"
        for i, item in enumerate(context)
    ])
    
    prompt = f"""You are RAKBank Service Navigation Assistant! You are here to help you find the correct login paths for various banking services on both web and mobile platforms for customers. Customers can ask you about any of the following services, and you will provide you with the precise steps to access them. Start with thinking steps inside <thinking></thinking> tag in a step-by-step way before formulating your final answer. Your final answer should be between <answer></answer> tag and in markdown only. If you cannot find the answer in the provide context, just say I don't have the answer between the answer tags only.

Question: {query}
DB Conversational Search Query Based on Past Conversation: {search_query}

Context:
{context_text}

Answer the question using only the provided context. Be honest and concise. """
    
    response = await models_and_data["async_openai_client"].chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
        stream=True
    )

    meta_response = {
        "query": query,
        "search_query": search_query,
        "context": context,
    }

    return StreamingResponse(stream_generator(response, meta_response),media_type='text/event-stream')



# @app.post("/rag_stream_2")
# async def rag(request: Conversations):
# conversations = request.conversations
# query = conversations[-1]['user']
# top_k = request.top_k

# # Create an async generator to yield streamed data 
# async def stream_results():
# # Handle query generation
# if len(conversations) == 1:
# search_query = conversations[0]['user']
# else:
# conversation_text = "\n".join([
# f"User: {msg['user']}\nAssistant: {msg.get('assistant', '')}"
# for msg in conversations
# ])
# prompt = f"""Given the following conversation, generate a focused search query that captures the main information need:

# Conversation:
# {conversation_text}

# Example: 
# change password with emirates
# block card
# replace card


# Generate a concise search query that best represents the user's current information need. Focus on the last user message to check if it is conversational or not. Do not change the query if it is not conversational in nature. Start with thinking steps inside <thinking></thinking> tag in step by step way before formulating you final search query. Provide your final search query inside <search_query></search_query> tag."""

            
# response = models_and_data["openai_client"].chat.completions.create(
# model=model_name,
# messages=[{"role": "user", "content": prompt}],
# temperature=0.3,
# max_tokens=1024,
# )
            
# llm_search_query = response.choices[0].message.content.strip()
# search_query = re.findall(r"<search_query>(.*?)</search_query>", llm_search_query, re.DOTALL)
# search_query = search_query[0] if search_query else query
        
# yield json.dumps({"phase": "query_generation", "data": search_query}) + "\n"
# await asyncio.sleep(0.1)
        
# # Simulate sentence embeddings for search query
# with torch.no_grad():
# q_em = models_and_data["all-minilm-l6-v2"].encode(search_query.lower())
# semantic_search_results = semantic_search(q_em, models_and_data['embeddings'], top_k=top_k)[0]

# # Stream search results part to client
# yield json.dumps({"phase": "semantic_search", "data": semantic_search_results}) + "\n"
# await asyncio.sleep(0.1)
        
# context = []
# for res in semantic_search_results:
# corpus_id = res['corpus_id']
# score = res['score']
# nearest_sentence = models_and_data['df'].row(corpus_id)[0]
# nearest_intent = models_and_data['df'].row(corpus_id)[1]
            
# if score > CONFIDENCE_THRESHOLD:
# context.append({
# "nearest_sentence": nearest_sentence,
# "nearest_intent": nearest_intent,
# "score": score,
# })

# yield json.dumps({"phase": "context_extraction", "data": context}) + "\n"
# await asyncio.sleep(0.1)

# if not context:
# yield json.dumps({
# "query": query,
# "search_query": search_query,
# "context": [],
# "answer": "I'm sorry, but there is no relevant information in the provided context to answer your question."
# }) + "\n"
# return

# context_text = "\n\n".join([
# f"Nearest Sentence {i+1}: {item['nearest_sentence']}\nNearest Intent {i+1}: {item['nearest_intent']}"
# for i, item in enumerate(context)
# ])
        
# prompt = f"""Identify the intent based on the following user query and context..."""

# response = models_and_data["openai_client"].chat.completions.create(
# model=model_name,
# messages=[{"role": "user", "content": prompt}],
# temperature=0.1,
# max_tokens=1024,
# )

# llm_answer = response.choices[0].message.content
# answer = re.findall(r"<answer>(.*?)</answer>", llm_answer, re.DOTALL)
# answer = answer[0].strip() if answer else llm_answer

# llm_english_answer = re.findall(r"<english_answer>(.*?)</english_answer>", llm_answer, re.DOTALL)
# english_answer = llm_english_answer[0] if llm_english_answer else ""
        
# final_response = {
# "query": query,
# "search_query": search_query,
# "context": context,
# "answer": answer,
# "english_answer": english_answer
# }

# yield json.dumps(final_response) + "\n"

# return StreamingResponse(stream_results(), media_type="application/json")