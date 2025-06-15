import os
import csv
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




dataset_path = '..\ACM_Workshop_SYSML\CIFAR10_dirichlet0.05_12'
yaml_pattern = os.path.join(dataset_path, '**/train_dataset_config.yaml')
yaml_files = glob.glob(yaml_pattern, recursive=True)
print(f"Found {len(yaml_files)} YAML files.")


# Output CSV file
output_csv = "client_label_distribution.csv"

# Prepare CSV header
labels = [f"label_{i}" for i in range(10)]
header = ["client_id", "num_items"] + labels

# Prepare list to collect all rows
rows = []
configurations = {}

# Traverse all subfolders (part_0 to part_11)
for subdir in os.listdir(dataset_path):
    part_path = os.path.join(dataset_path, subdir)
    if os.path.isdir(part_path):
        config_path = os.path.join(part_path, "CIFAR10_dirichlet0.05_12", "train_dataset_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)

            part_name = subdir
            metadata = data.get("metadata", {})
            label_dist = metadata.get("label_distribution", {})
            num_items = metadata.get("num_items", 0)
            
            # Save to configurations dict
            configurations[part_name] = metadata

            # Initialize all label columns with 0
            label_values = [0.0] * 10
            for k, v in label_dist.items():
                label_idx = int(k)
                label_values[label_idx] = v

            rows.append([subdir, num_items] + label_values)

# Write to CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"CSV file generated: {output_csv}")

"""
def load_yaml_files(yaml_files):
    configurations = {}
    for yaml_file in yaml_files:
        print(yaml_file)
        part_paths = yaml_file.split('/')
        part_name = None
        for part in part_paths:
            if part.startswith('part_'):
                part_name = part
                break

        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        configurations[part_name] = config['metadata']
    return configurations

configurations = load_yaml_files(yaml_files)
"""
print(f"Loaded configurations for {len(configurations)} parts.\n")

cifar_10_classes = {
    '0': 'airplane',
    '1': 'automobile',
    '2': 'bird',
    '3': 'cat',
    '4': 'deer',
    '5': 'dog',
    '6': 'frog',
    '7': 'horse',
    '8': 'ship',
    '9': 'truck'
}

label_distribution = []
for part_id, config in configurations.items():
    part_num = int(part_id.split('_')[1])
    
    for label, proportion in config['label_distribution'].items():
        label_distribution.append({
            'part_id': part_id,
            'part_num': part_num,
            'label': int(label),
            'class_name': cifar_10_classes[label],
            'proportion': proportion,
            'count': int(proportion * config['num_items'])
        })

label_df = pd.DataFrame(label_distribution)

# 1. Heatmap of Label Proportions Across Parts
plt.figure(figsize=(14, 10))

# Create pivot table for heatmap
pivot_proportions = label_df.pivot(index='part_num', columns='class_name', values='proportion')
pivot_proportions = pivot_proportions.fillna(0)

sns.heatmap(pivot_proportions, 
            annot=True, 
            fmt='.3f', 
            cmap='YlOrRd', 
            cbar_kws={'label': 'Proportion'})

plt.title('Label Distribution', 
          fontsize=16, fontweight='bold')
plt.xlabel('CIFAR-10 Classes', fontsize=12)
plt.ylabel('Part Number', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Heatmap.png", dpi=300)
plt.show()

print("Heatmap shows the proportion of each class in each part.")
print("Higher values (darker colors) indicate higher concentration of that class.\n")

# 2. Stacked Bar Chart of Absolute Counts
plt.figure(figsize=(15, 8))

# Create pivot table for counts
pivot_counts = label_df.pivot(index='part_num', columns='class_name', values='count')
pivot_counts = pivot_counts.fillna(0)

ax = pivot_counts.plot(kind='bar', stacked=True, figsize=(15, 8), 
                      colormap='tab10', width=0.8)

plt.title('Absolute Sample Counts by Class ', 
          fontsize=16, fontweight='bold')
plt.xlabel('Part Number', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.legend(title='CIFAR-10 Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("Barchart.png", dpi=300)
plt.show()

# Display summary statistics
print("Summary of sample counts per part:")
total_samples_per_part = pivot_counts.sum(axis=1)
print(total_samples_per_part.describe())
