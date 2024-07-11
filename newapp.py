from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings 
from Load_Data import MongoDBHandler
from datetime import datetime  
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  
from langchain_openai import OpenAIEmbeddings  
from langchain_community.chat_models import ChatOpenAI  
import os
import bson
from pymongo import MongoClient
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import streamlit as st

st.set_page_config(page_title="Financial Advisor Chatbot", page_icon="💬")

# Define the SimpleDocument class
class SimpleDocument:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

    def __str__(self):
        metadata_str = ', '.join(f'{key}: {value}' for key, value in self.metadata.items())
        return f"Text: {self.page_content}\nMetadata: {metadata_str}"

os.environ["OPENAI_API_KEY"] = 'sk-proj-p8hPKdKoqa47EfoL9rseT3BlbkFJeGMEqlVBHvcYEUzRYZdR'

mongo_handler = MongoDBHandler()

def fetch_documents(doc_type, email=None):
    return mongo_handler.get_info(doc_type, email)

def get_text_in_brackets(text):
    start_bracket = text.rfind('[')
    end_bracket = text.rfind(']')
    if start_bracket != -1 and end_bracket != -1 and start_bracket < end_bracket:
        return text[start_bracket + 1:end_bracket].strip()
    return ''

def uploadDataToMongo(question, answer, type, mongo_url='mongodb://localhost:27017/', db_name='demo', collection_name='interactions'):
    client = MongoClient(mongo_url)
    db = client[db_name]
    collection = db[collection_name]
    document = {
        "type": type,
        "question": question,
        "answer": answer
    }
    result = collection.insert_one(document)
    print("Chat uploaded to MongoDB", result)

def remove_brackets(text):
    start_bracket = text.rfind('[')
    if start_bracket != -1:
        return text[:start_bracket].rstrip()
    return text

def split_text(text, chunk_size=1500, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


def create_documents(info, doc_type):
    documents = []
    if doc_type == "transactions":
        text = ""
        for transaction in info["transactions"]:
            newText = (
                f"Date: {transaction['date'].strftime('%Y-%m-%d') if isinstance(transaction['date'], datetime) else 'N/A'}, "
                f"Category: {transaction.get('category', 'N/A')}, "
                f"Amount: {transaction.get('amount', 'N/A')}, "
                f"Type: {transaction.get('type', 'N/A')}, "
                f"Transaction ID: {transaction.get('transaction_id', 'N/A')}, "
                f"Account ID: {transaction.get('account_id', 'N/A')}, "
                f"Description: {transaction.get('description', 'N/A')}\n"
            )
            text = text + newText
        metadata = {"type": "transaction"}
        documents.append(SimpleDocument(text, metadata))
    elif doc_type == "investments":
        text = f"Initial Investments: {info['initial_investments']}\n"
        text += "Investment Performance:\n"
        for performance in info["investment_performance"]:
            text += f"Month: {performance['month']}\n"
            for investment, amount in performance.items():
                if investment != "month":
                    text += f"  - {investment}: {amount}\n"
        metadata = {"type": "investment"}
        documents.append(SimpleDocument(text, metadata))
    elif doc_type == "information":
        text = info.get("content", "")
        metadata = {"type": "information"}
        for chunk in split_text(text):
            documents.append(SimpleDocument(chunk, metadata))
    return documents

email = "Jamesjoyce123@gmail.com"
doc_types = ["transactions", "investments", "information"]
all_documents = []

for doc_type in doc_types:
    try:
        if doc_type == "information":
            info = fetch_documents(doc_type) 
        else:
            info = fetch_documents(doc_type, email)
    
        documents = create_documents(info, doc_type)
        all_documents.extend(documents)
    except ValueError as e:
        print(e)

for doc in all_documents:
    print()
    print(doc)
embedding = OpenAIEmbeddings()
persist_directory = 'db/chroma/'

vectordb = Chroma.from_documents(
    documents=all_documents,
    embedding=embedding,
    persist_directory=persist_directory
)

# Initialize the language model
llm_name = "gpt-3.5-turbo"
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
a single word to the end of your answer [Savings, Investment Plan, Spending Habits, Discovery Card Plans, Plot, Other]
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

st.title("Financial Advisor Chatbot 💬")

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
    print(response['result'])
    typeText = get_text_in_brackets(response['result'])
    uploadDataToMongo(user_question, response['result'], type=typeText)  # to store feedback
    output_string = remove_brackets(response['result'])
    
    if typeText == "Plot":
        st.session_state.show_plot = True
        st.session_state.messages.append({"role": "assistant", "type": "image", "content": "plot.png"})
    else:
        st.session_state.show_plot = False
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": output_string})
    
    st.experimental_rerun()

#if st.session_state.show_plot:
#    st.image("plot.png", caption="Account Balance Over Time")
