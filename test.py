from langchain_openai import AzureOpenAIEmbeddings
# from sentence_transformers import SentenceTransformer, util

embeddings = AzureOpenAIEmbeddings(
    deployment="text-data-002",  # Azure OpenAI deployment name
    model="text-embedding-ada-002",  # Model you want to use
    api_key="db8d369a30e840b39ccdfdce4808ec7f",  # Your Azure OpenAI API key
    azure_endpoint="https://rakbankgenaidevai.openai.azure.com/",  # Your Azure OpenAI resource URL
    openai_api_version="2023-05-15"  # API version you're using
)

# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# query = "its benefits?"
# context = "What is world elite credit card?"

# query = "Can you explain how backpropagation works?"
# context = "Can you tell me more about training neural networks? Training a neural network involves adjusting the weights of connections between neurons based on error reduction. This is typically done through a process called backpropagation, which calculates the error for each layer and adjusts weights to minimize it. The process repeats for multiple epochs until the model reaches optimal accuracy."

query = "How long are the skywards miles valid for? "
context = "What is the cashback anywhere program?"

query_embedding = embeddings.embed_query(context)
query_embedding1 = embeddings.embed_query(query)


# query_embedding = embedding_model.encode(query, convert_to_tensor=True)
# context_embedding = embedding_model.encode(context, convert_to_tensor=True)
# x = util.cos_sim(query_embedding, context_embedding).item()
# print(x)

import openai
from scipy.spatial.distance import cosine

def calculate_similarity_openai(query, context):
    query_embedding = embeddings.embed_query(query)
    context_embedding = embeddings.embed_query(context)
    # Cosine similarity calculation
    similarity = 1 - cosine(query_embedding, context_embedding)
    return similarity

# print(calculate_similarity_openai(query, context))

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# def tfidf_similarity(query, context):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform([query, context])
#     similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
#     return similarity[0][0]

# print(tfidf_similarity(query, context))
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# def calculate_similarity_tfidf(query, context):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform([query, context])
#     similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
#     return similarity

# print(calculate_similarity_tfidf(query, context))

# from gensim.models import KeyedVectors
# from gensim.similarities import WmdSimilarity

# # Load pre-trained embeddings
# # word_vectors = KeyedVectors.load_word2vec_format("path/to/GoogleNews-vectors-negative300.bin", binary=True)
# # instance = WmdSimilarity([context], word_vectors, num_best=1)

# # def calculate_similarity_wmd(query, context):
# #     # Compute WMD similarity
# #     similarity_score = 1 - instance[query][0][1]  # Invert distance to get similarity
# #     return similarity_score

# # print(calculate_similarity_wmd(query, context))

# from transformers import BertTokenizer, BertModel
# import torch
# from scipy.spatial.distance import cosine

# # Load BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# def get_cls_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     outputs = model(**inputs)
#     cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # CLS token
#     return cls_embedding



# def calculate_similarity_bert(query, context):
#     query_embedding = get_cls_embedding(query)
#     context_embedding = get_cls_embedding(context)
#     similarity = 1 - cosine(query_embedding.detach().numpy(), context_embedding.detach().numpy())
#     return similarity

# print(calculate_similarity_bert(query, context))

# # from sklearn.decomposition import TruncatedSVD
# # from sklearn.feature_extraction.text import TfidfVectorizer

# # def calculate_similarity_lsa(query, context):
# #     vectorizer = TfidfVectorizer()
# #     tfidf_matrix = vectorizer.fit_transform([query, context])
    
# #     # Decompose TF-IDF matrix using SVD
# #     svd = TruncatedSVD(n_components=100)
# #     lsa_matrix = svd.fit_transform(tfidf_matrix)
    
# #     # Calculate cosine similarity
# #     similarity = cosine_similarity(lsa_matrix[0:1], lsa_matrix[1:2])[0][0]
# #     return similarity


# # print(calculate_similarity_lsa(query, context))


# Mock embeddings to simulate the functionality
def mock_embed_query(query):
    embeddings_dict = {
        "What are the minimum monthly charges for not maintaining minimum balance in a savings account?": embeddings.embed_query("What are the minimum monthly charges for not maintaining minimum balance in a savings account?"),
        "What about for RakBooster?": embeddings.embed_query("What about for RakBooster?"),
        "And for a current account?": embeddings.embed_query("And for a current account?"),
        "Just in AED?": embeddings.embed_query("Just in AED?"),
        "How long are Skywards miles valid for?": embeddings.embed_query("How long are Skywards miles valid for?"),
    }
    return embeddings_dict.get(query, [0, 0])

def cosine_similarity(v1, v2):
    # Cosine similarity function
    return 1 - cosine(v1, v2)


def is_followup_single(query, last_query, threshold=0.7):
    query_embedding = mock_embed_query(query)
    last_embedding = mock_embed_query(last_query)
    similarity = cosine_similarity(query_embedding, last_embedding)
    return similarity >= threshold

import numpy as np

def calculate_weighted_average(embeddings, weights):
    weighted_embeddings = [np.array(emb) * weight for emb, weight in zip(embeddings, weights)]
    return np.sum(weighted_embeddings, axis=0) / sum(weights)

def is_followup_weighted(query, previous_queries, threshold=0.7, min_similarity=0.5, decay_factor=0.7):
    query_embedding = mock_embed_query(query)
    all_embeddings = [mock_embed_query(q) for q in previous_queries]
    
    # Exponential decay weights (more recent queries have higher weight)
    weights = [decay_factor ** i for i in range(len(all_embeddings))][::-1]
    context_embedding = calculate_weighted_average(all_embeddings, weights)
    
    similarity = cosine_similarity(query_embedding, context_embedding)
    if similarity < min_similarity:
        return False
    return similarity >= threshold

conversation = [
    "What are the minimum monthly charges for not maintaining minimum balance in a savings account?",
    "What about for RakBooster?",
    "And for a current account?",
    "Just in AED?",
    "How long are Skywards miles valid for?"
]

# Initialize results
results_single = []
results_weighted = []

# # Test single-last-query approach
# for i in range(1, len(conversation)):
#     results_single.append(
#         (conversation[i], is_followup_single(conversation[i], conversation[i - 1]))
#     )

# # Test weighted-average approach
# for i in range(1, len(conversation)):
#     results_weighted.append(
#         (conversation[i], is_followup_weighted(conversation[i], conversation[:i]))
#     )

# # Print results
# print("Single-Last-Query Method Results:")
# for query, is_followup in results_single:
#     print(f"Query: '{query}' -> Follow-up: {is_followup}")

# print("\nWeighted-Average Method Results:")
# for query, is_followup in results_weighted:
#     print(f"Query: '{query}' -> Follow-up: {is_followup}")


from scipy.spatial.distance import cosine
import numpy as np

def calculate_similarity_openai(query, recent_contexts, threshold=0.75):
    # Generate embedding for the current query
    query_embedding = embeddings.embed_query(query)

    # Generate embeddings for recent contexts and average them
    context_embeddings = [embeddings.embed_query(context) for context in recent_contexts]
    avg_context_embedding = np.mean(context_embeddings, axis=0)
    
    # Calculate cosine similarity
    # similarity = 1 - cosine(query_embedding, avg_context_embedding)
    def cosine_similarity(A, B):
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    # Calculate similarity between the two embeddings
    similarity = cosine_similarity(query_embedding, avg_context_embedding)

    
    # Classify as follow-up if similarity exceeds threshold
    is_followup = similarity >= threshold
    return similarity, is_followup
# query = "How much does a liability letter cost ?"                                                                     #Not well
# recent_contexts=["Customer is making a salary of 10000 AED a month are the eligible for skyworld credit card?", "So how much should I make?","is 50000 AED fine?"]

# query = "So how much should I make?"
# recent_contexts=["Customer is making a salary of 10000 AED a month are the eligible for skyworld credit card?"]

# query = "How much does a liability letter cost ?"
# recent_contexts = ["Customer is making a salary of 10000 AED a month are the eligible for skyworld credit card?"]

# query = "Can non-rakbank customers open it?"
# recent_contexts = ["What is the eligibility for a Gold Account?"]

# query = "What is the policy for the physical redemption of gold?"
# recent_contexts = ["What is the eligibility for a Gold Account?"]                                                 #Not well

# query = "Who is elon musk?"
# recent_contexts = ["What is the eligibility for a Gold Account?"]

# query = "What are the key features of the Savings Account?"
# recent_contexts = ["What is the eligibility for a Gold Account?"]                                                 #Not well

query = "What are the key features of the Savings Account?"
recent_contexts = ["What is the eligibility for a Gold Account?","Can non-RAKBANK customers open a gold account?"]  #Not well

sim, follw = calculate_similarity_openai(query,recent_contexts)
print("cosine_similarity::",sim, follw)