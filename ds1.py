import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
csv_file_path = r"C:\Users\Hp\Downloads\archive\retail_sales_dataset.csv"  # Use raw string
df = pd.read_csv(csv_file_path, parse_dates=['Date'])

# Summary Statistics
print("Dataset Overview:")
print(df.describe())

# Ensure 'Date' is set as an index for time series analysis
df.set_index('Date', inplace=True)

# Sales Trend Over Time 
plt.figure(figsize=(14, 6))

# Aggregate daily sales
daily_sales = df['Total Amount'].resample('D').sum()

# Plot bar graph
daily_sales.plot(kind='bar', color='skyblue', width=0.8)

# Formatting for better readability
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Sales (in Rupees)', fontsize=12)
plt.title('Daily Sales Trend Over Time', fontsize=14)
plt.xticks(ticks=range(0, len(daily_sales), max(1, len(daily_sales) // 10)), 
           labels=daily_sales.index.strftime('%Y-%m-%d')[::max(1, len(daily_sales) // 10)], 
           rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Reset index for further analysis
df.reset_index(inplace=True)

# Sales by Product Category & Gender (Modified Second Graph)
plt.figure(figsize=(12, 6))
sns.barplot(x='Product Category', y='Total Amount', hue='Gender', data=df, estimator=np.sum, ci=None)

# Formatting
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Total Sales (in Rupees)', fontsize=12)
plt.title('Total Sales by Product Category and Gender', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.show()

# Age Distribution of Customers (More Precise)
plt.figure(figsize=(8, 4))
sns.histplot(df['Age'], bins=range(15, 70, 5), kde=True)  # Precise binning for 5-year intervals
plt.xlabel('Age Group (Years)', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)  # Added precise y-axis title
plt.title('Age Distribution of Customers', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Sales Distribution by Age Group (Pie Chart)
# Define age bins
age_bins = list(range(15, 75, 5))  # Creating bins of 5 years each
age_labels = [f"{age_bins[i]}-{age_bins[i+1]}" for i in range(len(age_bins) - 1)]  # e.g., "20-25", "25-30"
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Aggregate sales by age group
age_sales = df.groupby('Age Group')['Total Amount'].sum()

# Plot pie chart
plt.figure(figsize=(8, 8))
colors = sns.color_palette('pastel')
plt.pie(age_sales, labels=age_sales.index, autopct='%1.1f%%', colors=colors, startangle=140)
plt.title('Sales Contribution by Age Group', fontsize=14)
plt.show()

# Create Age Group column
age_bins = list(range(15, 75, 5))
age_labels = [f"{age_bins[i]}-{age_bins[i+1]}" for i in range(len(age_bins) - 1)]
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Pivot table for relation heatmap: Product Category vs Age Group (Total Sales)
relation_pivot = df.pivot_table(
    index='Product Category',
    columns='Age Group',
    values='Total Amount',
    aggfunc='sum',
    fill_value=0
)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(relation_pivot, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=0.5)
plt.title('Sales Relation: Product Category vs Age Group', fontsize=14)
plt.xlabel('Age Group')
plt.ylabel('Product Category')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
