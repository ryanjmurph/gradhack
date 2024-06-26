import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
os.environ['OPENAI_API_KEY']='sk-proj-p8hPKdKoqa47EfoL9rseT3BlbkFJeGMEqlVBHvcYEUzRYZdR' 

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
  db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
  db = SQLDatabase.from_uri(db_uri)
  return db

def get_sql_chain(db):
  template = """
	You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For examples:
    Question:Can you provide the first name, last name, and balance for each customer?
    SQL Query: SELECT Customers.id AS CustomerID, Customers.first_name AS FirstName, Customers.last_name AS LastName, Account.balance AS Balance FROM Customers JOIN Account ON Customers.id = Account.customer_id;
    Question: "Can you show the loan details along with the account balance and customer name for each loan?
    SQL Query: SELECT Loan.id AS LoanID, Loan.amount_paid AS AmountPaid, Loan.start_date AS StartDate, Loan.due_date AS DueDate, Account.id AS AccountID, Account.balance AS AccountBalance, Customers.first_name AS CustomerFirstName, Customers.last_name AS CustomerLastName FROM
    Loan JOIN Account ON Loan.account_id = Account.id JOIN Customers ON Account.customer_id = Customers.id;
    Question:Can you provide details of each transaction, including the account balance and the customer's first and last names?
    SQL Query: SELECT Transaction.id AS TransactionID, Transaction.description AS Description, Transaction.amount AS Amount, Transaction.date AS Date, Account.id AS AccountID, Account.balance AS AccountBalance, Customers.first_name AS CustomerFirstName, Customers.last_name AS CustomerLastName
    FROM Transaction JOIN Account ON Transaction.account_id = Account.id JOIN Customers ON Account.customer_id = Customers.id;
	End examples
    Your turn: 
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  llm = ChatOpenAI()
  def get_schema(_):
    return db.get_table_info()

  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
    Based on the schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatOpenAI()
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a personal banking assistant. Ask me anything about your personal financing."),
    ]



st.set_page_config(page_title="Chat with your bank", page_icon=":speech_balloon:")

st.title("Chat with bank")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="Dadsname2", key="Password")
    st.text_input("Database", value="Bank", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))

