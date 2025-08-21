import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv("Groceries_dataset.csv")
grouped_transactions = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).tolist()

encoder = TransactionEncoder()
encoded_transactions = encoder.fit_transform(grouped_transactions)

transaction_df = pd.DataFrame(encoded_transactions, columns=encoder.columns_)

frequent_itemsets = apriori(transaction_df, min_support=0.005, use_colnames=True)
association_rule_df = association_rules(frequent_itemsets, metric="lift", min_threshold=0.3)

frequent_itemsets
association_rule_df
