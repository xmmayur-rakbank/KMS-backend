from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",  # Azure OpenAI deployment name
    model="text-embedding-ada-002",  # Model you want to use
    api_key="643406f5cf994e82bf1dadecf0e120f1",  # Your Azure OpenAI API key
    azure_endpoint="https://gpt-demo-openai.openai.azure.com/",  # Your Azure OpenAI resource URL
    openai_api_version="2023-12-01-preview"  # API version you're using
)

async def deleting(filename, username):
    try:
        persist_local_directory="chroma_db"
        chroma_db_instance = Chroma(persist_directory=persist_local_directory, embedding_function=embeddings)
        print(chroma_db_instance.get())
        
        all_docs = chroma_db_instance.get()
        x=all_docs['metadatas']
        y=all_docs['ids']

        indexlist=[]
        
        for index, obj in enumerate(all_docs['metadatas']):
            print(f"Index: {index}, Object: {obj}")
            file=obj["filename"]
            if(obj["filename"]==filename):
                indexlist.append(index)

        # for index in indexlist:
        #     del 
        indexlistids=[]
        for index,value in enumerate(all_docs['ids']):
            if(index  in indexlist):
                indexlistids.append(value )

        chroma_db_instance.delete(ids=indexlistids)       
        # Filter documents that match the metadata condition (filename = 'XZ')
        # docs_to_delete = [doc for doc in all_docs if doc.metadata.get("filename") == "XZ"]

        # # Extract their IDs
        # doc_ids_to_delete = [doc.id for doc in docs_to_delete]

        # # Delete documents by their IDs
        # chroma_db_instance.delete(ids=doc_ids_to_delete)

        # metadata_filter = {"filename": filename}

        # metadata_filter = {"department": "HR"}

        # Query the Chroma database to fetch documents based on the metadata filter
        # results = chroma_db_instance.get(include=["documents", "metadatas"], where=metadata_filter)

        # Extract documents, ids, and metadata
        # documents = results['documents']
        # document_ids = results['ids']
        # metadata = results['metadatas']

        # Print the results
        # for doc_id, doc, meta in zip(document_ids, documents, metadata):
        #     print(f"Document ID: {doc_id}")
        #     print(f"Document: {doc}")
        #     print(f"Metadata: {meta}")
        #     print("-----")
        # print(chroma_db_instance.get())
        # # Delete documents with HR metadata
        # # chroma_db_instance.delete(where=metadata_filter)
        # ids_to_delete = []

        # for doc in chroma_db_instance.get():
        #     if doc.metadata.get('filename') == filename:
        #         ids_to_delete.append(doc.id)

        # chroma_db_instance.delete(ids=ids_to_delete)
        return f'{username} deleted file {filename} successfully.'
    except Exception as e:
        print(e)
