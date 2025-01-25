from langchain.schema import Document
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
import json
import os, pymongo, pprint
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

load_dotenv()

@tool
def CustomerSupport() -> str:
    """Information about customer support"""
    return """call customer support at 1-800-555-1234"""

tools = [CustomerSupport]
chat = ChatGroq(temperature=0.2, model_name="llama3-70b-8192").bind_tools(tools)
json_data = json.load(open("senseacademia_20241206.json"))
# Define the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase for better context
    chunk_overlap=200
)

# Function to split values and create Document objects
def create_documents_from_json(json_obj, max_chunk_size):
    documents = []
    for key, value in json_obj.items():
        if isinstance(value, str) and len(value) > max_chunk_size:
            # Split long strings and create Document objects
            chunks = text_splitter.split_text(value)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"Link": key}))
        else:
            # Add non-splittable values as Document objects
            documents.append(Document(page_content=value, metadata={"key": key}))
    return documents

# Create Document objects from JSON
documents = create_documents_from_json(json_data, max_chunk_size=20)

# Create the vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)




prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the official AI assistant for Sense Academia, a professional training and educational institution. Your role is to provide accurate, helpful information about Sense Academia's courses, programs, and services while maintaining a professional and supportive tone."),
    ("user", """You will be given context and a query. 
                {context}
                {query}
                
                If the context is relevant to the query, provide a detailed response based on the context. 
                Do not call any tools.

                If the context is not relevant, call the CustomerSupport() tool to get the response.

                Example 1: 
                Query: What is the weather today? 
                Response: Call CustomerSupport() tool (as the context is not relevant).

                Example 2: 
                Query: What Internet of Things courses are available? 
                Response: Provide a detailed response based on the context (as the context is relevant)."""),
])


# print((vectorstore.similarity_search_with_score("hello",k=3)[0]))
def process_data(data):
    response = []
    docs = vectorstore.similarity_search_with_score(data,k=3)
    for doc in docs:
        response.append({"context":doc[0].page_content,"metadata":doc[0].metadata})
    # print(response)
    return response
    
# Chain the prompt and model
chain = {"context":lambda query:process_data(query), "query":lambda query:query}|prompt |chat

# Example usage
response = chain.invoke("Who is the best cricket in the world?")
print(response)