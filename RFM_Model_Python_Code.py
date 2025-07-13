import os
import subprocess
import zipfile

# Set your working directory
os.chdir("C:/Users/prati/PycharmProjects/PythonProject/draft")  # ✅ update if needed

# Download the dataset using subprocess
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "jihyeseo/online-retail-data-set-from-uci-ml-repo"
], check=True)

# Unzip the downloaded file
with zipfile.ZipFile("online-retail-data-set-from-uci-ml-repo.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

print("Dataset downloaded and extracted successfully!")

import pandas as pd

# Load the dataset
df = pd.read_excel("Online Retail.xlsx")

# View the first 5 rows
print(df.head())


import pandas as pd

# Load data
df = pd.read_excel("Online Retail.xlsx")

# 1. Drop rows where CustomerID is missing
df = df[df['CustomerID'].notnull()]

# 2. Remove cancelled orders (InvoiceNo starts with 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# 3. Filter for United Kingdom transactions only
df = df[df['Country'] == 'United Kingdom']

# 4. Create a TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Preview cleaned data
print("Cleaned data shape:", df.shape)
print(df.head())

import datetime as dt

# Snapshot date — the day after the last invoice date
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# Group by CustomerID to calculate RFM metrics
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                    # Frequency
    'TotalPrice': 'sum'                                        # Monetary
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Preview the RFM table
print("RFM table created:")
print(rfm.head())

# Score each metric from 1 (worst) to 4 (best)
rfm['R'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])  # Lower recency = better
rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])

# Combine scores into one RFM score
rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

# Preview the scored data
print("RFM scores assigned:")
print(rfm[['Recency', 'Frequency', 'Monetary', 'R', 'F', 'M', 'RFM_Score']].head())

# Export RFM table to Excel
rfm.to_excel("RFM_Segmentation.xlsx")

print("RFM table exported to Excel successfully!")


# Define customer segments based on RFM_Score
def segment_customer(score):
    if score >= '444':
        return 'Best Customers'
    elif score[0] == '4':
        return 'Recent Customers'
    elif score[1] == '4':
        return 'Frequent Buyers'
    elif score[2] == '4':
        return 'Big Spenders'
    elif score <= '111':
        return 'At Risk'
    else:
        return 'Others'

# Apply segmentation
rfm['Segment'] = rfm['RFM_Score'].apply(segment_customer)

# View some of the results
print(" Customer Segments:")
print(rfm[['RFM_Score', 'Segment']].head())


import matplotlib.pyplot as plt

# Count of customers in each segment
segment_counts = rfm['Segment'].value_counts()

# Bar chart
plt.figure(figsize=(10, 6))
segment_counts.plot(kind='bar')
plt.title('Customer Segments by RFM Score')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Reset index so CustomerID becomes a column
rfm_final = rfm.reset_index()

# Export the RFM table with segmentation to Excel
rfm_final.to_excel("C:/Users/prati/PycharmProjects/PythonProject/draft/RFM_Output.xlsx", index=False)

print("RFM model exported to Excel.")
