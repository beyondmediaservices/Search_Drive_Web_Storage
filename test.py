## Sample code to create a tool , create an agent and deploy it successfully 

import os
from google.cloud import aiplatform
import vertexai
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import VectorSearchVectorStore
from langchain_googledrive.retrievers import GoogleDriveRetriever
from langchain_google_community import GoogleDriveLoader
from langchain_google_community import GCSDirectoryLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
import json
import argparse


from langchain_community.vectorstores import FAISS
from langchain_google_vertexai.llms import VertexAI
from vertexai.preview import reasoning_engines
from dotenv import load_dotenv
load_dotenv()

# Set your GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Vertex_AI_Key.json"
LOCATION =  os.environ.get('LOCATION')
GDRIVE_FOLDER_ID = os.environ.get('GDRIVE_FOLDER_ID')
PROJECT_ID = os.environ.get('PROJECT_ID')
GOOGLE_ACCOUNT_FILE = os.environ.get('GOOGLE_ACCOUNT_FILE')
MODEL =  os.environ.get('MODEL')
GCS_BUCKET=os.environ.get('GCS_BUCKET')

llm = VertexAI(model_name=MODEL)
embeddings_vertex = VertexAIEmbeddings(model_name="text-embedding-004")
retriever = None


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def search(query: str, retriever) -> str:

    # This will give mutliple results
    retrieval_qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
    )
    result = retrieval_qa_with_sources.invoke(query, return_only_outputs=True)
    return result
    """

    # This will only give a final output
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
    )
    result = retrieval_qa.invoke(query, return_only_outputs=True)
    return result
    """


## --- Vertex AI and Retriever --- ######
def create_vertex_ai_vector_as_retriever(documents, vector_db_path="faiss_db"):
    faiss_vector_database = FAISS.from_documents(documents, embeddings_vertex)
    faiss_vector_database.save_local(vector_db_path)

    retriever = faiss_vector_database.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={'score_threshold': 0.1, "k": 3}
    )

    return retriever

# --- Document Loaders ---
def load_documents_from_gcs(GCS_BUCKET):
    """Loads documents from a Google Bucket."""
    loader = GCSDirectoryLoader(
        bucket=GCS_BUCKET,
        project_name=PROJECT_ID,
    )
    #.load() only read the documents from GCS
    documents = loader.load_and_split()
    retriever = create_vertex_ai_vector_as_retriever(documents)
    return documents

def load_documents_from_google_drive(GDRIVE_FOLDER_ID):
    """Loads documents from a Google Drive folder."""
    loader = GoogleDriveLoader(
        folder_id=GDRIVE_FOLDER_ID,
        service_account_key=".credentials/keys.json",
        credentials_path=".credentials/credentials.json",
        recursive=True,
        template="gdrive-all-in-folder",
        scopes= ["https://www.googleapis.com/auth/drive"]
    )
    #.load() only read the documents from Drive
    #documents = loader.load()

    #After reading the documents, we need to split in chunks
    #load_and_split load as well as split those into chunks by default use RecursiveCharacterTextSplitter return list[Document]
    documents = loader.load_and_split()
    return documents

# --- Document Loaders ---

def search_GDRIVE():
    documents = load_documents_from_google_drive(GDRIVE_FOLDER_ID)
    retriever = create_vertex_ai_vector_as_retriever(documents)
    #Getting data by quering Retriever
    response = search("Medal Tally of Namish", retriever)
    res_dict = {}
    res_dict["source_documents"] = [] 

    for each_source in response["source_documents"]:
            res_dict["source_documents"].append({
                "page_content": each_source.page_content,
                "metadata":  each_source.metadata
            })

    print(json.dumps(res_dict["source_documents"], indent=4, default=str))


def search_GCS():
    documents = load_documents_from_gcs(GCS_BUCKET)
    retriever = create_vertex_ai_vector_as_retriever(documents)

    #Getting data by quering Retriever
    response = search("What is holistic developement of a Child", retriever)
    res_dict = {}
    res_dict["source_documents"] = [] 

    for each_source in response["source_documents"]:
            res_dict["source_documents"].append({
                "page_content": each_source.page_content,
                "metadata":  each_source.metadata
            })

    print(json.dumps(res_dict["source_documents"], indent=4, default=str))

def search_ALL():
    # We will search from both Drive and GCS, 
    # Here document loader is loading and then splitting the docs in chunks
    documents_GCS = load_documents_from_gcs(GCS_BUCKET)
    documents_DRIVE = load_documents_from_google_drive(GDRIVE_FOLDER_ID)
    # Combine the two sets of split documents
    combined_split_docs = documents_GCS + documents_DRIVE
    retriever = create_vertex_ai_vector_as_retriever(combined_split_docs)

    #Getting data by quering Retriever
    #We can use any s_GCS or s_DRIVE here in comnined search

    response = search("What are kapil's Hobbies", retriever)
    res_dict = {}
    res_dict["source_documents"] = [] 

    for each_source in response["source_documents"]:
            res_dict["source_documents"].append({
                "page_content": each_source.page_content,
                "metadata":  each_source.metadata
            })

    print(json.dumps(res_dict["source_documents"], indent=4, default=str))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This is a simple example script")
    parser.add_argument("-s", "--search", help = "Show Output")

    # Parse the arguments
    args = parser.parse_args()

    if args.search == "GCS":
        search_GCS()
    elif args.search == "GDRIVE":
        search_GDRIVE()
    else:
        search_ALL()
        
    """
    # --- Create Reasoning Engines Agents ---
    research_agent = reasoning_engines.LangchainAgent(
        model_kwargs={"temperature": 0},
        tools=[search],
        agent_executor_kwargs={"return_intermediate_steps": True},
        model=MODEL,
    )
    #Now you can test the model and agent behavior to ensure that it's working as expected before you deploy it:
    research_query = "What is holistic developement of a Child"
    response = research_agent.query(
        input=research_query
    )
    print(response['output'])
    """
