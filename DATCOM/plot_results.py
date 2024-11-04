# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read the file and skip the first line, then overwrite the file
with open("for042.csv", "r") as file:
    lines = file.readlines()

# Write back all lines except the first one
with open("for042_no_first_line.csv", "w") as file:
    file.writelines(lines[1:])

df = pd.read_csv("for042_no_first_line.csv")

# Remove single quotes around each column name
df.columns = df.columns.str.replace("'", "").str.strip()

df


print(df.columns.tolist())
#metrics = [...]  # Initialize metrics with a list of metrics
metrics = df.columns.tolist()
variables_to_drop = ['CASE', 'RAD?', 'TRIM?', 'MACH', 'Unnamed: 52']
metrics = list(filter(lambda metric: metric not in variables_to_drop, metrics))
metrics
#if metrics is None:
#    metrics = []  # Initialize it to an empty list if it is None
#metrics = list(filter(lambda metric: metric not in variables_to_drop, metrics))
#print(f"metrics before filtering: {metrics}")  # Check the value of metrics


# Print unique values in the 'ALPHA' column
print(df['ALPHA'].unique())


# Print unique values in the 'MACH' column
print(df['MACH'].unique())


def plot_individual_metrics_vs_mach(df, metrics=None, save_path='plots/'):
    # Set default metrics if none provided
    if metrics is None:
        metrics = ['RE', 'ALT', 'Q', 'BETA', 'PHI', 'SREF', 'XCG', 'XMRP', 'LREF', 'LATREF', 'ALPHA', 'CN', 'CM', 'CA', 'CA_0B', 'CA_FB', 'CY', 'CLN', 'CLL', 'CL', 'CD', 'CL/CD', 'X-C.P.', 'CNA', 'CMA', 'CYB', 'CLNB', 'CLLB', 'CNQ', 'CMQ', 'CAQ', 'CNAD', 'CMAD', 'CYQ', 'CLNQ', 'CLLQ', 'CYR', 'CLNR', 'CLLR', 'CYP', 'CLNP', 'CLLP', 'CNP', 'CMP', 'CAP', 'CNR', 'CMR', 'CAR']
    
    # Loop through each metric to create separate figures
    for metric in metrics:
        print(metric)
                    
        plt.figure(figsize=(12, 8))
        
        # Loop through each unique ALPHA value
        for alpha_value in df['ALPHA'].unique():
            # Filter DataFrame for the current ALPHA value
            df_filtered = df[df['ALPHA'] == alpha_value]
            
            # Plot the metric against MACH for the current ALPHA value
            plt.plot(df_filtered['MACH'], df_filtered[metric], marker='o', linestyle='-', label=f'ALPHA = {alpha_value}')
        
        plt.xlabel('MACH')
        plt.ylabel(metric)
        plt.title(f'{metric} vs MACH for Different ALPHA Values')
        plt.legend()
        plt.grid(True)

        # Save the figure in different formats
        plt.savefig(f'{save_path}{metric}_vs_MACH.png', format='png')
        plt.savefig(f'{save_path}{metric}_vs_MACH.eps', format='eps')
        plt.savefig(f'{save_path}{metric}_vs_MACH.pdf', format='pdf')
            
        plt.show()

columns_to_write = ['MACH', 'ALPHA', 'CA', 'CN', 'CM']
df.to_csv('MACH_ALPHA_CA_CN_CM.csv', columns=columns_to_write, index=False)

# Usage
plot_individual_metrics_vs_mach(df)



