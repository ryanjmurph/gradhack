import json
from pymongo import MongoClient
from bson import json_util
import matplotlib.pyplot as plt
from datetime import datetime

class Plotter:
    def __init__(self, db_name="demo", collection_name="discovery"):
        # Connect to MongoDB (adjust the connection string as needed)
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def plot_account_balance(self, email, save_path=None):
        # Retrieve the document for the specified email
        user_document = self.collection.find_one({"email": email})

        if not user_document:
            raise ValueError(f"No user found with email: {email}")

        # Extract relevant details
        opening_amount = user_document["opening_amount"]
        monthly_income = user_document["monthly_income"]
        transactions = user_document["transactions"]

        if not transactions:
            raise ValueError("No transactions found for the user.")

        # Initialize variables for time series data
        dates = []
        balances = []

        # Determine the initial date from the earliest transaction date
        initial_date = min(transaction["date"] for transaction in transactions)
        current_balance = opening_amount + monthly_income

        # Add the initial balance to the time series data
        dates.append(initial_date)
        balances.append(current_balance)

        # Process each transaction and update the balance
        for transaction in sorted(transactions, key=lambda x: x["date"]):
            current_balance -= transaction["amount"]
            dates.append(transaction["date"])
            balances.append(current_balance)

        # Plot the time series graph
        plt.figure(figsize=(10, 6))
        plt.plot(dates, balances, marker='o', linestyle='-', color='b')
        plt.xlabel('Date')
        plt.ylabel('Balance (R)')
        plt.title('Account Balance Over Time')
        plt.xticks(rotation=45)
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

# Example usage:
# Create an instance of AccountBalancePlotter
plotter = Plotter()

# Plot the account balance and save the plot to a file without top and right spines
plot_result = plotter.plot_account_balance("Jamesjoyce123@gmail.com", save_path="plot.png")
print(plot_result)

# Or simply display the plot without top and right spines
plot_result = plotter.plot_account_balance("Jamesjoyce123@gmail.com")
print(plot_result)


