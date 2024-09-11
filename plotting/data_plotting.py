import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator

if __name__ == '__main__':
    file_path = 'Subgraphs_entities_per_step.csv'
    file_path = os.path.join("plotting", file_path)
    # Read the data from the CSV file
    data = pd.read_csv(file_path, delimiter=';')

    # Extract columns from the dataframe
    x = data['entities_per_step']
    sampling_time = data['sampling_time']
    triples_deviation = data['triples_deviation']

    # Create the plot
    fig, ax = plt.subplots(2, 1, figsize=(9, 7))

    # Plot sampling_time on the primary y-axis (left)
    color = 'tab:blue'
    ax[0].set_xlabel('Entities per Sampling Step')
    ax[0].set_ylabel('Sampling Time [s]', color=color)
    ax[0].plot(x, sampling_time, color=color, marker='d')
    ax[0].tick_params(axis='y', labelcolor=color)

    # Add light grey grid lines for the primary y-axis and x-axis
    ax[0].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='lightgrey')

    color = 'tab:red'
    ax[1].set_xlabel('Entities per Sampling Step')
    ax[1].set_ylabel('Deviation of triples [# triples]', color=color)
    ax[1].plot(x, triples_deviation, color=color, marker='d')
    ax[1].tick_params(axis='y', labelcolor=color)
    ax[1].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='lightgrey')

    # Add regression
    slope, intercept = np.polyfit(x, triples_deviation, 1)
    regression_line = slope * x + intercept
    ax[1].plot(x, regression_line, color='lightcoral', linestyle='-.')
    # Show the plot
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------------------------------------------

    # Define the file path
    file_path = 'Subgraph_amount.csv'
    file_path = os.path.join("plotting", file_path)

    # Load the data
    data = pd.read_csv(file_path, delimiter=';')

    # Get unique Subgraph_amount values
    x = data['Subgraph_amount'].unique()

    # Initialize lists to store MRR values for each rho
    mrr_values = []
    rho_values = [-1, 0.5, 1, 2]

    # Collect MRR values for each rho
    for rho in rho_values:
        filtered_data = data.loc[data['rho'] == rho, ['Subgraph_amount', 'MRR']]
        # Sort by Subgraph_amount to maintain order in the plot
        filtered_data = filtered_data.sort_values(by='Subgraph_amount')
        mrr_values.append(filtered_data['MRR'].values)

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    ax.set_xlabel('Subgraph amount N')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('MRR [%]')

    # Plot MRR values for each rho
    for mrr_rho, rho in zip(mrr_values, rho_values):
        ax.plot(x, mrr_rho, label=f'rho={rho}')

    # Add legend and show plot
    ax.legend(loc='best')
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='lightgrey')
    plt.show()
