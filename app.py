# Import necessary libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI  # import openai
import os
import bson
from pymongo import MongoClient
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from emailer import Email
from streamlit_feedback import streamlit_feedback
from plotter import Plotter

# Streamlit functions
def close_chat():
    st.write("chat session closed")

# Set Streamlit page configuration
st.set_page_config(page_title="Financial Advisor Chatbot", page_icon="💬")


# Function to load data from MongoDB
def loadDataFromMongo(mongo_url='mongodb://localhost:27017/', db_name='gradhack', collection_name='discovery'):
    client = MongoClient(mongo_url)
    db = client[db_name]
    collection = db[collection_name]
    documents = collection.find()
    data = list(documents)
    client.close()
    return data

def uploadDataToMongo(question, answer, category, mongo_url='mongodb://localhost:27017/', db_name='gradhack', collection_name='interactions'):
    client = MongoClient(mongo_url)
    db = client[db_name]
    collection = db[collection_name]
    document = {
        "category": category,
        "question": question,
        "answer": answer
    }
    result = collection.insert_one(document)
    print("Chat uploaded to MongoDB", result)
    client.close()


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

# Helper function to remove square bracket category from response
def remove_brackets(text):
    start_bracket = text.rfind('[')
    if start_bracket != -1:
        return text[:start_bracket].rstrip()
    return text

# Helper function to get text in brackets
def get_text_in_brackets(text):
    start_bracket = text.rfind('[')
    end_bracket = text.rfind(']')
    if start_bracket != -1 and end_bracket != -1 and start_bracket < end_bracket:
        return text[start_bracket + 1:end_bracket].strip()
    return ''

# Load data from MongoDB
data = loadDataFromMongo()
print(data)

# Extract transactions from the content
def extract_transactions(content):
    transactions = []
    for line in content.split('\n'):
        if 'Spent' in line or 'Deposited' in line or 'Withdrew' in line or 'Refunded' in line:
            transactions.append(line.strip())
    return transactions

transactions = []
docs = []
for doc in data:
    if 'content' in doc:  # Assuming 'content' field contains the text data
        filtered_metadata = filter_complex_metadata(doc)
        docs.append(SimpleDocument(text=doc['content'], metadata=filtered_metadata))
        transactions.extend(extract_transactions(doc['content']))
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
                transactions.append(f"{transaction.get('date', 'N/A')}: {transaction.get('amount', 'N/A')} - {transaction.get('description', 'N/A')}")
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
persist_directory = 'ddb/chroma/'

# Create new vector store if no embeddings are already stored
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
# print(vectordb._collection.count())

# Load the vector store if embeddings are already stored
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Initialize the language model
llm_name = "gpt-4o"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# Define the prompt template for the QA chain
template = """
You are a highly knowledgeable and reliable financial advisor who works for Discovery Bank. 
Use the provided context to answer the questions accurately and professionally.
Try to give personal advice to people. Try to find the information in the tables 
to base your advice off of and if you cant find it then you can give generic advice. 
When recommending cards or accounts, please only recommend discovery cards and accounts, not made up ones. 
Only answer questions if they pertain to finance and only include answers that have something to do with finance.
If the context does not contain enough information or you are unsure of the answer, clearly state that you don't know.
Three sentence responses are enough. 
When referring to James Joyce, please always use information from his account as context.
Try to mention Vitality Points when asked about their financial situation
Please talk to me as if I am James Joyce. So when I ask you questions pretend you are talking to James Joyce
and use his information as mine.
When I ask questions about how James Joyce can cut down on expenses please refer to his account transactions and 
identify expenses that are non-essential like parties or coffees or dining out.
Never make up information that does not exist in the db.
Always prioritize the user's financial well-being and avoid making assumptions.
Always elaborate in your answers and tell the user why you came up with the conclusion you came up with.
Always answer me like you are giving me advice not general advice. Like I am getting personal advice from
my financial advisor. Use words like "you".
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:

At the end of each answer categorize the topic of the question into one of the following topics and add this topic as
a single word to the end of your answer [Savings, Investment Plan, Spending Habits, Discovery Card Plans,
Transaction Plot, Investment Plot, Other] 
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

logo_path = "logo.png"  
st.sidebar.image(logo_path, width=100)
st.title("Financial Advisor Chatbot 💬", )
transactions = """15th Jan 2024: -R100 on groceries\n
10th Feb 2024: Deposited R500 (paycheck)\n
10th Feb 2024: -R150 on dinner\n
12th Feb 2024: -R50 at a party\n
15th Feb 2024: -R40 at a coffee shop\n
17th Feb 2024: -R40 at a coffee shop\n
18th Feb 2024: -R67 at a party\n
20th Feb 2024: -R67 at a party\n
24th Feb 2024: -R40 at a coffee shop\n
1st Mar 2024: -R25 on dinner\n
2nd Mar 2024: -R75 on BP petrol\n
3rd Mar 2024: -R200 from an ATM\n
10th Mar 2024: Refunded R150 for clothing\n
15th Mar 2024: -R100 at a birthday party\n
20th Mar 2024: -R150 at a dinner party\n
21st Mar 2024: -R200 at a party\n
31st Oct 2024: -R250 at a party"""

st.sidebar.subheader("Transactions")
st.sidebar.write(transactions)

if st.sidebar.button("Email"):
    user_question = """Give me an overall run down of my finances, 
    how im doing in reaching my savings goals. Also mention my investments in stocks and how they have done.
    Write a paragraph for the finance rundown and another for the investments
    Do not mention my investments in the first paragraph just my transactions.
    In the second paragprah only mention my investments
     Write as much as you can.
    """
    response = qa_chain({"query": user_question})
    email_client = Email()
    print(response['result'], " THis is the result")

    plotter = Plotter()

    plot_result_investments = plotter.plot_investments("Jamesjoyce123@gmail.com", save_path="investments_plot.png")
    print(plot_result_investments)

    plot_result_balance = plotter.plot_account_balance("Jamesjoyce123@gmail.com", save_path="balance_plot.png")
    print(plot_result_balance)

    image_paths = ['investments_plot.png', 'balance_plot.png']

    email_client.send_email(response['result'], image_paths)

# Close chat functions

# instantiate session variables
if 'rating' not in st.session_state:
    st.session_state.rating = 1
if 'feedback' not in st.session_state:
    st.session_state.feedback = ""

# Function to upload back to mongo
def uploadFeedback(mongo_url='mongodb://localhost:27017/', db_name='gradhack', collection_name='feedback'):
    client = MongoClient(mongo_url)
    db = client[db_name]
    collection = db[collection_name]
    document = {
        "rating": st.session_state.rating,
        "comment": st.session_state.feedback
    }
    result = collection.insert_one(document)
    print("Feedback uploaded to MongoDB", result)
    client.close()


if st.sidebar.button("Close Chat"):
    st.sidebar.write("Thank you for using the chat")
    with st.sidebar.form("feedback_form"):
        st.write("we would love to hear your feedback")
        st.slider("Rate your experience",1,10, key = 'rating')
        st.text_area("Any Comments or suggestions?", key = 'feedback')
        st.form_submit_button("Submit", on_click = uploadFeedback)



if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "type": "text", "content": "How can I help you?"}]
if "show_plot" not in st.session_state:
    st.session_state["show_plot"] = False

def display_chat_message(role, msg_type, content):
    if role == "assistant":
        col1, col2 = st.columns([1, 9])
        with col1:
            st.image(logo_path, width=40)
        with col2:
            if msg_type == "text":
                st.markdown(content)
            elif msg_type == "image":
                st.image(content)
    else:
        if msg_type == "text":
            st.chat_message(role).write(content)
        elif msg_type == "image":
            st.image(content)

for msg in st.session_state.messages:
    display_chat_message(msg["role"], msg["type"], msg["content"])

if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})
    st.experimental_rerun()  # Force the app to rerun so the user's message appears immediately

if "messages" in st.session_state and st.session_state.messages[-1]["role"] == "user":
    user_question = st.session_state.messages[-1]["content"]
    response = qa_chain({"query": user_question})
    print(response['result'], "RESULT")
    typeText = get_text_in_brackets(response['result'])
    uploadDataToMongo(user_question, response['result'], category = typeText)  # to store feedback
    output_string = remove_brackets(response['result'])
    
    if typeText == "Transaction Plot":
        st.session_state.show_plot = True
        st.session_state.messages.append({"role": "assistant", "type": "image", "content": "balance_plot.png"})
    elif typeText == "Investment Plot":
        st.session_state.show_plot = True
        st.session_state.messages.append({"role": "assistant", "type": "image", "content": "investments_plot.png"})   
    else:
        st.session_state.show_plot = False
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": output_string})
    
    st.experimental_rerun()


#if st.session_state.show_plot:
#    st.image("plot.png", caption="Account Balance Over Time")