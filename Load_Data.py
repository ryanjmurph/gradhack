from pymongo import MongoClient
from datetime import datetime

class MongoDBHandler:
    def __init__(self, mongo_url='mongodb://localhost:27017/', db_name='demo', collection_name='discovery'):
        self.client = MongoClient(mongo_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def get_user_document(self, doc_type, email=None):
        query = {"type": doc_type}
        if email:
            query["email"] = email
        user_document = self.collection.find_one(query)
        if not user_document:
            if email:
                raise ValueError(f"No document found with email: {email} and type: {doc_type}")
            else:
                raise ValueError(f"No document found with type: {doc_type}")
        return user_document

    def extract_transaction_details(self, user_document):
        transactions = user_document.get("transactions", [])
        for transaction in transactions:
            transaction["transaction_id"] = transaction.get("transaction_id", "N/A")
            transaction["account_id"] = transaction.get("account_id", "N/A")
            transaction["description"] = transaction.get("description", "N/A")
            transaction["category"] = transaction.get("category", "N/A")
        details = {
            "name": user_document.get("name", "N/A"),
            "email": user_document.get("email", "N/A"),
            "opening_amount": user_document.get("opening_amount", "N/A"),
            "monthly_income": user_document.get("monthly_income", "N/A"),
            "transactions": self.convert_dates(transactions),
            "closing_balance": user_document.get("closing_balance", "N/A")
        }
        return details

    def extract_investment_details(self, user_document):
        details = {
            "name": user_document.get("name", "N/A"),
            "email": user_document.get("email", "N/A"),
            "initial_investments": user_document.get("initial_investments", {}),
            "investment_performance": user_document.get("investment_performance", [])
        }
        return details

    def extract_information_details(self, user_document):
        details = {
            "content": user_document.get("content", "")
        }
        return details

    def convert_dates(self, transactions):
        for transaction in transactions:
            if isinstance(transaction["date"], dict) and "$date" in transaction["date"]:
                transaction["date"] = datetime.strptime(transaction["date"]["$date"], '%Y-%m-%dT%H:%M:%S.%fZ')
        return transactions

    def get_info(self, doc_type, email=None):
        user_document = self.get_user_document(doc_type, email)
        if doc_type == "transactions":
            details = self.extract_transaction_details(user_document)
            return details
        elif doc_type == "investments":
            details = self.extract_investment_details(user_document)
            return details
        elif doc_type == "information":
            details = self.extract_information_details(user_document)
            return details
        else:
            raise ValueError(f"Invalid document type: {doc_type}")
