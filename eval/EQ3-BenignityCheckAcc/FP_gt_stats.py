import pandas as pd
import csv

# Read the CSV file
df = pd.read_csv('./FP_ground_truth.csv')

# Count the values in the 'is_FP?' column
fp_counts = df['is_FP?'].value_counts()
yes_count = fp_counts.get('Yes', 0)  # True Positives (packages correctly identified as FP)
no_count = fp_counts.get('No', 0)    # False Positives (packages incorrectly identified as FP)

# Count missing values
missing_values = df['is_FP?'].isna().sum()
unknown_values = 0
if 'Unkonwn' in fp_counts:  # Note the typo in "Unkonwn"
    unknown_values = fp_counts['Unkonwn']

# Count unavailable packages
unavailable = df[df['Comment'] == 'Unavailable package'].shape[0]
unavailable += df[df['Comment'] == 'Unavailable'].shape[0]

# Print the results
print(f"True Positives (Yes): {yes_count}")
print(f"False Positives (No): {no_count}")
print(f"Unknown: {unknown_values}")
print(f"Missing values: {missing_values}")
print(f"Unavailable packages: {unavailable}")

# Calculate percentages
total = len(df)
print(f"\nTotal entries: {total}")
print(f"Yes percentage: {yes_count/total*100:.2f}%")
print(f"No percentage: {no_count/total*100:.2f}%")
print(f"Unknown/Missing percentage: {(unknown_values+missing_values)/total*100:.2f}%")

# Additional analysis - count by FP Categories
if 'FP Categories (Rules)' in df.columns:
    categories = df['FP Categories (Rules)'].dropna().str.split(',').explode().str.strip()
    category_counts = categories.value_counts()
    print("\nFP Categories breakdown:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")
