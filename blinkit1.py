# ==============================
# 📦 Importing Required Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (8, 5)

# ==============================
# 📥 Load Dataset
# ==============================
df = pd.read_csv("blinkit_dataset.csv")

# ==============================
# 🧹 Data Overview & Cleaning
# ==============================
print("Shape of dataset:", df.shape)
print("\nColumns:", list(df.columns))
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
print("\nData Info:\n")
print(df.info())
print("\nSummary Statistics:\n", df.describe())

# Fill missing values
df['Item_Name'] = df['Item_Name'].fillna('Unknown_Item')
df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])
df['Outlet_Type'] = df['Outlet_Type'].fillna(df['Outlet_Type'].mode()[0])

# ==============================
# 📊 Data Visualization Section
# ==============================

# 1️⃣ Distribution of Item MRP
plt.figure()
sns.histplot(df['Item_MRP (₹)'], bins=25, kde=True, color='teal')
plt.title('Distribution of Item MRP')
plt.xlabel('Item MRP (₹)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 2️⃣ Distribution of Item Outlet Sales
plt.figure()
sns.histplot(df['Item_Outlet_Sales (₹)'], bins=25, kde=True, color='orange')
plt.title('Distribution of Item Outlet Sales')
plt.xlabel('Item Outlet Sales (₹)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 3️⃣ Relationship: Item MRP vs Sales
plt.figure()
sns.scatterplot(
    x='Item_MRP (₹)',
    y='Item_Outlet_Sales (₹)',
    data=df,
    alpha=0.6,
    color='green'
)
plt.title('Item MRP vs Item Outlet Sales')
plt.xlabel('Item MRP (₹)')
plt.ylabel('Item Outlet Sales (₹)')
plt.tight_layout()
plt.show()

# 4️⃣ Average Sales by Outlet Type
plt.figure()
sns.barplot(
    x='Outlet_Type',
    y='Item_Outlet_Sales (₹)',
    data=df,
    estimator=np.mean,
    palette='viridis'
)
plt.title('Average Sales by Outlet Type')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 5️⃣ Sales Distribution by Outlet Size & Type
plt.figure()
sns.boxplot(
    x='Outlet_Size',
    y='Item_Outlet_Sales (₹)',
    hue='Outlet_Type',
    data=df,
    palette='Set2'
)
plt.title('Sales Distribution by Outlet Size and Type')
plt.xlabel('Outlet Size')
plt.ylabel('Item Outlet Sales (₹)')
plt.legend(title='Outlet Type', loc='upper right')
plt.tight_layout()
plt.show()

# 6️⃣ Total Sales by Item Type (Descending)
plt.figure()
item_sales = (
    df.groupby('Item_Type')['Item_Outlet_Sales (₹)']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
sns.barplot(
    x='Item_Type',
    y='Item_Outlet_Sales (₹)',
    data=item_sales,
    palette='muted'
)
plt.title('Total Sales by Item Category (Descending Order)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 7️⃣ Correlation Matrix of Numeric Columns
plt.figure(figsize=(10,8))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.show()

# ==============================
# 🕒 Sales Trend Over Time
# ==============================

# If no date column exists, generate synthetic dates for trend analysis
if 'Date' not in df.columns:
    df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')

# Convert and group by month
df['Date'] = pd.to_datetime(df['Date'])
monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Item_Outlet_Sales (₹)'].sum()

# Line chart for monthly trend
monthly_sales.index = monthly_sales.index.to_timestamp()
plt.figure()
monthly_sales.plot(marker='o', color='purple')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales (₹)')
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# 📈 Top 10 Outlets by Average Sales
# ==============================
plt.figure()
top_outlets = (
    df.groupby('Outlet_Identifier')['Item_Outlet_Sales (₹)']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
sns.barplot(
    x=top_outlets.index,
    y=top_outlets.values,
    palette='cool'
)
plt.title('Top 10 Outlets by Average Sales')
plt.xlabel('Outlet Identifier')
plt.ylabel('Average Sales (₹)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nBlinkit Dataset Analysis Completed Successfully!")
