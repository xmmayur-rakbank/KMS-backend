from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",  # Azure OpenAI deployment name
    model="text-embedding-ada-002",  # Model you want to use
    api_key="643406f5cf994e82bf1dadecf0e120f1",  # Your Azure OpenAI API key
    azure_endpoint="https://gpt-demo-openai.openai.azure.com/",  # Your Azure OpenAI resource URL
    openai_api_version="2023-05-15"  # API version you're using
)
import psycopg2

async def feedbackSave(SessionId, MessageId, Answer, Question, Textual_Feedback, Numerical_Feedback):
    try:
        print(SessionId,MessageId, Answer, Question, Textual_Feedback, Numerical_Feedback)
        db_config = {
            "host": "kms-postgres.postgres.database.azure.com",
            "dbname": "feedback",
            "user": "kmsadmin",
            "password": "Celebal@123456",
            "port": 5432,  # Default PostgreSQL port
            "sslmode": "require",  # Enforce SSL
        }

        create_table_query = """
        CREATE TABLE IF NOT EXISTS KMSfeedback (
            id SERIAL PRIMARY KEY,
            SessionId INT NOT NULL,
            MessageId INT NOT NULL,
            Answer TEXT NOT NULL,
            Question TEXT NOT NULL,
            Textual_Feedback TEXT NOT NULL,
            Numerical_Feedback INT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        try:
            # Connect to PostgreSQL
            with psycopg2.connect(**db_config) as conn:
                with conn.cursor() as cur:
                    # Create feedback table
                    cur.execute(create_table_query)
                    print("Feedback table created successfully.")
        except psycopg2.Error as e:
            print(f"An error occurred: {e}")
            
        feedback_data = {
            "SessionId": SessionId,
            "MessageId": MessageId,
            "Answer": Answer,
            "Question": Question,
            "Textual_Feedback":Textual_Feedback,
            "Numerical_Feedback":Numerical_Feedback
        }

        # SQL to insert feedback
        insert_query = """
        INSERT INTO KMSfeedback (SessionId, MessageId, Answer, Question, Textual_Feedback, Numerical_Feedback)
        VALUES (%s, %s, %s, %s, %s, %s);
        """

        try:
            # Connect to PostgreSQL
            with psycopg2.connect(**db_config) as conn:
                with conn.cursor() as cur:
                    # Insert feedback data
                    cur.execute(insert_query, (feedback_data["SessionId"], feedback_data["MessageId"], feedback_data["Answer"], feedback_data["Question"], feedback_data["Textual_Feedback"], feedback_data["Numerical_Feedback"]))
                    conn.commit()  # Commit transaction
                    print("Feedback inserted successfully.")
                    return "Feedback inserted successfully."
        except psycopg2.Error as e:
            print(f"An error occurred: {e}")
        return True
    except Exception as e:
        print(e)


    try:
        # persist_local_directory="chroma_db"
        persist_local_directory="./chroma_langchain_db"
        # chroma_db_instance = Chroma(persist_directory=persist_local_directory, embedding_function=embeddings)
        chroma_db_instance = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db_v1",  # Where to save data locally, remove if not necessary
        )
        
        # persist_directory="./chroma_langchain_db",
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
        File_set = set()
        for index, obj in enumerate(all_docs['metadatas']):
            print(f"Index: {index}, Object: {obj}")
            file=obj["filename"]
            department=obj["department"]
            if(file==filename):
                Finallist.append((index))
        list1=[]
        for index, obj in enumerate(all_docs['documents']):
            if(index in Finallist):
                list1.append(obj)
        # list1=chroma_db_instance.get_by_ids(Finallist)
        print(list1)
        return list1
    except Exception as e:
        print(e)