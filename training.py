from langchain_openai import AzureOpenAIEmbeddings
import shutil
import os
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import uuid


embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",  # Azure OpenAI deployment name
    model="text-embedding-ada-002",  # Model you want to use
    api_key="643406f5cf994e82bf1dadecf0e120f1",  # Your Azure OpenAI API key
    azure_endpoint="https://gpt-demo-openai.openai.azure.com/",  # Your Azure OpenAI resource URL
    openai_api_version="2023-12-01-preview"  # API version you're using
)

async def training(file,departmant):
    unique_id=uuid.uuid4()
    temp_dir=create_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path=save_file_to_temp_directory(file, temp_dir)
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load_and_split()
    for doc in documents:
         doc.metadata['department'] = departmant
         doc.metadata['filename'] = str(file.filename)
    
    print("Document info",len(documents))
    print("Document details",(documents))
    docs=split_docs(documents)
    persist_local_directory=os.path.join(departmant+ "/" + str(file.filename))
    print("Chunking completed for document info",len(docs))

    # chroma_db_instance=Chroma.from_documents(docs, embeddings, persist_directory=persist_local_directory)

    
    # Assuming `docs` is the new set of documents you want to add
    def add_documents_to_chroma(docs, department, persist_local_directory):
        # Load existing ChromaDB
        chroma_db_instance = Chroma(persist_directory=persist_local_directory, embedding_function=embeddings)
        
        # Add metadata to documents (e.g., department info)
        for doc in docs:
            doc.metadata["department"] = department
            doc.metadata["filename"] = str(file.filename)
        # Add new documents to the existing Chroma instance
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        chroma_db_instance.add_documents(docs, ids=doc_ids)
        
        # Persist changes to the database
        # chroma_db_instance.persist()

    # Usage
    persist_local_directory = "chroma_db"
    add_documents_to_chroma(docs, departmant, persist_local_directory)

    
    print("chromadb created successfully.")
    
    # text = "this is a test document"
    # query_result = embeddings.embed_query(text)
    # doc_result = embeddings.embed_documents([text])
    # print(doc_result[0][:5])

def split_docs(documents, chunk_size=700, chunk_overlap=120):
     if documents is None:
          print("No documents to split")
          return None
     splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
     chunks=splitter.split_documents(documents)
     return chunks
     
def create_temp_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        print("temp dir:",temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def save_file_to_temp_directory(file, temp_dir):
    
        temp_file_path=os.path.join(temp_dir, file.filename)
        with open(temp_file_path, 'wb') as f:
            shutil.copyfileobj(file.file,f)
    
        return temp_file_path