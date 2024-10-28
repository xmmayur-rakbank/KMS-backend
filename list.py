from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",  # Azure OpenAI deployment name
    model="text-embedding-ada-002",  # Model you want to use
    api_key="643406f5cf994e82bf1dadecf0e120f1",  # Your Azure OpenAI API key
    azure_endpoint="https://gpt-demo-openai.openai.azure.com/",  # Your Azure OpenAI resource URL
    openai_api_version="2023-12-01-preview"  # API version you're using
)

async def listing():
    try:
        persist_local_directory="chroma_db"
        chroma_db_instance = Chroma(persist_directory=persist_local_directory, embedding_function=embeddings)
        print(chroma_db_instance.get())
        
        all_docs = chroma_db_instance.get()
        x=all_docs['metadatas']
        y=all_docs['ids']

        indexlist=[]
        # set = {""}
        # Departmentset = {""}
        Finallist = []
        Finallist = []
        unique_pairs = set()
        
        for index, obj in enumerate(all_docs['metadatas']):
            print(f"Index: {index}, Object: {obj}")
            file=obj["filename"]
            department=obj["department"]
            # set.add(file)
            # Departmentset.add(department)
            Finallist.append([file,department])
            pair = (file, department)
    
            # Check if the pair is already in the set
            if pair not in unique_pairs:
                unique_pairs.add(pair)  # Add to the set of unique pairs
                Finallist.append([file, department])
        
        return Finallist
    except Exception as e:
        print(e)
