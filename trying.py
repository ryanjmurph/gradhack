# Import necessary libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import os
import bson
from pymongo import MongoClient
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import streamlit as st

# Function to load data from MongoDB
def loadDataFromMongo(mongo_url='mongodb://localhost:27017/', db_name='gradhack', collection_name='discovery'):
    client = MongoClient(mongo_url)
    db = client[db_name]
    collection = db[collection_name]
    documents = collection.find()
    data = list(documents)
    return data

# Define the SimpleDocument class
class SimpleDocument:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

# Utility function to filter complex metadata types
def filter_complex_metadata(metadata):
    filtered_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            filtered_metadata[key] = value
        elif isinstance(value, bson.ObjectId):
            filtered_metadata[key] = str(value)  # Convert ObjectId to string
    return filtered_metadata

# Load data from MongoDB
data = loadDataFromMongo()

docs = []
for doc in data:
    if 'content' in doc:  # Assuming 'content' field contains the text data
        filtered_metadata = filter_complex_metadata(doc)
        docs.append(SimpleDocument(text=doc['content'], metadata=filtered_metadata))
    else:
        # Process structured client data
        content = f"Client Name: {doc.get('name', 'N/A')}\n"
        content += f"Client Age: {doc.get('age', 'N/A')}\n"
        content += f"Client Address: {doc.get('address', 'N/A')}\n"
        if 'accounts' in doc:
            for account in doc['accounts']:
                content += f"Account ID: {account.get('account_id', 'N/A')}, Type: {account.get('account_type', 'N/A')}, Open Balance: {account.get('open_balance', 'N/A')}, Balance: {account.get('balance', 'N/A')}, Ending Balance: {account.get('ending_balance', 'N/A')}\n"
        if 'transactions' in doc:
            for transaction in doc['transactions']:
                content += f"Transaction ID: {transaction.get('transaction_id', 'N/A')}, Account ID: {transaction.get('account_id', 'N/A')}, Date: {transaction.get('date', 'N/A')}, Amount: {transaction.get('amount', 'N/A')}, Description: {transaction.get('description', 'N/A')}\n"
        filtered_metadata = filter_complex_metadata(doc)
        docs.append(SimpleDocument(text=content, metadata=filtered_metadata))

# Text splitter to handle large documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

# Create splits of the documents using the text splitter
splits = text_splitter.split_documents(docs)

# Initialize the OpenAI API key
os.environ["OPENAI_API_KEY"] = 'sk-proj-p8hPKdKoqa47EfoL9rseT3BlbkFJeGMEqlVBHvcYEUzRYZdR' 

# Create the vector store
embedding = OpenAIEmbeddings()
persist_directory = 'docs/chroma/'

# Uncomment the following lines if you need to create the vector store for the first time
# vectordb = Chroma.from_documents(
#     documents=splits,
#     embedding=embedding,
#     persist_directory=persist_directory
# )
# print(vectordb._collection.count())

# Load the vector store if embeddings are already stored
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
print(vectordb._collection.count())

# Initialize the language model
llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# Define the prompt template for the QA chain
template = """
You are a highly knowledgeable and reliable financial advisor who works for Discovery Bank. 
Use the provided context to answer the questions accurately and professionally.
If the context does not contain enough information or you are unsure of the answer, clearly state that you don't know.
Three sentence responses are enough. 
Always prioritize the user's financial well-being and avoid making assumptions.
Always elborate in your answers and tell the user why you came up with the conclusion you came up with.
Always answer me like you are giving me advice not general advice. Like I am getting personal advice from
my financial advisor. Use words like "you".
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

# Initialize Streamlit session state for memory
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)

# Update memory in session state to use in the chain
memory = st.session_state.memory

# Create the RetrievalQA chain with context and history retention
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectordb.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": memory,
    }
)

# Streamlit UI
st.title("Financial Advisor Chatbot")

# Input box for the user's question
user_question = st.text_input("Ask a financial question:")

if st.button("Get Answer"):
    if user_question:
        result = qa_chain({"query": user_question})
        st.write("Answer:", result['result'])
        if 'source_documents' in result:
            st.write("Source Document:", result['source_documents'][0].page_content)
    else:
        st.write("Please enter a question.")


