import json
from pymongo import MongoClient
from bson import json_util
import matplotlib.pyplot as plt
from datetime import datetime

class Plotter:
    def __init__(self, db_name="gradhack", collection_name="discovery"):
        # Connect to MongoDB (adjust the connection string as needed)
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def plot_investments(self, email, save_path=None):
        user_document = self.collection.find_one({"email": email, "type": "investments"})

        if not user_document:
            raise ValueError(f"No investment data found for email: {email}")

        print("Retrieved document:", json.dumps(user_document, indent=2, default=str))

        investment_performance = user_document.get("investment_performance", [])

        if not investment_performance:
            raise ValueError("No investment performance data found for the user.")

        stock_data = {}

        for entry in investment_performance:
            date = datetime.strptime(entry["month"], "%Y-%m")
            for stock, value in entry.items():
                if stock == "month":
                    continue
                if stock not in stock_data:
                    stock_data[stock] = {"dates": [], "values": []}
                stock_data[stock]["dates"].append(date)
                stock_data[stock]["values"].append(value)

        plt.figure(figsize=(12, 8))
        for stock, data in stock_data.items():
            plt.plot(data["dates"], data["values"], marker='o', linestyle='-', label=stock)

        plt.xlabel('Date')
        plt.ylabel('Value (R)')
        plt.title('Investment Performance Over Time')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        # Remove the top and right spines (bounding box lines)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if save_path:
            # Save the plot to a file
            plt.savefig(save_path)
            plt.close()
            return f"Plot saved as {save_path}"
        else:
            # Show the plot
            plt.show()
            return "Plot displayed"

    def plot_account_balance(self, email, save_path=None):
        user_document = self.collection.find_one({"email": email, "type": "transactions"})

        if not user_document:
            raise ValueError(f"No transaction data found for email: {email}")

        opening_amount = user_document["opening_amount"]
        monthly_income = user_document["monthly_income"]
        transactions = user_document["transactions"]

        if not transactions:
            raise ValueError("No transactions found for the user.")

        dates = []
        balances = []

        initial_date = min(datetime.strptime(transaction["date"]["$date"], "%Y-%m-%dT%H:%M:%S.%fZ") if isinstance(transaction["date"], dict) else transaction["date"] for transaction in transactions)
        current_balance = opening_amount + monthly_income

        dates.append(initial_date)
        balances.append(current_balance)

        for transaction in sorted(transactions, key=lambda x: datetime.strptime(x["date"]["$date"], "%Y-%m-%dT%H:%M:%S.%fZ") if isinstance(x["date"], dict) else x["date"]):
            current_balance -= transaction["amount"]
            date = datetime.strptime(transaction["date"]["$date"], "%Y-%m-%dT%H:%M:%S.%fZ") if isinstance(transaction["date"], dict) else transaction["date"]
            dates.append(date)
            balances.append(current_balance)

        plt.figure(figsize=(10, 6))
        plt.plot(dates, balances, marker='o', linestyle='-', color='b')
        plt.xlabel('Date')
        plt.ylabel('Balance (R)')
        plt.title('Account Balance Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if save_path:
            plt.savefig(save_path)
            plt.close()
            return f"Plot saved as {save_path}"
        else:
            plt.show()
            return "Plot displayed"




