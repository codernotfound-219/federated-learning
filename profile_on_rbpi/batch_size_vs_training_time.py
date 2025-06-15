import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV data
df = pd.read_csv('batch_size_ablation.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nUnique batch sizes: {sorted(df['batch_size'].unique())}")

# Display first few rows
print("First 10 rows of the dataset:")
print(df.head(10))

print("\n" + "="*50 + "\n")

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

print("\n" + "="*50 + "\n")

# Basic statistics
print("Basic statistics:")
print(df.describe())

# Group by batch size to see training time statistics
print("Training time statistics by batch size:")
batch_stats = df.groupby('batch_size')['training_time'].agg(['count', 'mean', 'std', 'min', 'max'])
print(batch_stats.round(2))

# 4. Plot Batch Size vs Training Time
# Now let's create various visualizations to understand the relationship between batch size and training time.

# Create a boxplot to show training time distribution by batch size
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='batch_size', y='training_time')
plt.title('Training Time Distribution by Batch Size', fontsize=16, fontweight='bold')
plt.xlabel('Batch Size', fontsize=14)
plt.ylabel('Training Time (seconds)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Create a violin plot to show distribution shape
plt.figure(figsize=(12, 8))
sns.violinplot(data=df, x='batch_size', y='training_time')
plt.title('Training Time Distribution Shape by Batch Size', fontsize=16, fontweight='bold')
plt.xlabel('Batch Size', fontsize=14)
plt.ylabel('Training Time (seconds)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Create a bar plot showing average training time by batch size
avg_training_time = df.groupby('batch_size')['training_time'].mean().reset_index()

plt.figure(figsize=(10, 6))
bars = plt.bar(avg_training_time['batch_size'], avg_training_time['training_time'], 
               color=['skyblue', 'lightcoral', 'lightgreen'])

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')

plt.title('Average Training Time by Batch Size', fontsize=16, fontweight='bold')
plt.xlabel('Batch Size', fontsize=14)
plt.ylabel('Average Training Time (seconds)', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Analyze training time by client and batch size
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='batch_size', y='training_time', hue='client_id')
plt.title('Training Time Distribution by Batch Size and Client', fontsize=16, fontweight='bold')
plt.xlabel('Batch Size', fontsize=14)
plt.ylabel('Training Time (seconds)', fontsize=14)
plt.legend(title='Client ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

if __name__ == "__main__":
    print("Batch Size vs Training Time Analysis completed successfully!")
