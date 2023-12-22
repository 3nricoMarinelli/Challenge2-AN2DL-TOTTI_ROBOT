import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def thePlotter(categories, valid_periods, training_data):
    colors = sns.color_palette("pastel", 6)
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 20), sharex=False)
    for i, category in enumerate(np.unique(categories)):
      indices = np.where(categories==category)[0]
      random_indices = np.random.choice(indices, size=5, replace=False)
      start_series = np.min(valid_periods[random_indices][:,0])
      end_series = np.max(valid_periods[random_indices][:,0])
      for j in range(5):
        axes[i].plot(np.arange(start_series, training_data.shape[1]),training_data[random_indices[j], start_series:], color=colors[j], fillstyle="full")
        axes[i].set_ylabel('Value')
        axes[i].set_xlabel('Time')
        axes[i].set_title(category)
    fig.subplots_adjust(hspace=1.0)
    plt.show()

def the_statistician(valid_periods, categories, less_than, higher_than):
    print(f'Total number of samples: {valid_periods.shape[0]}')
    for i in np.unique(categories):
        mask = np.where(categories == i, True, False)
        print(f'Number of samples of category {i}: {np.sum(mask)}')
    period_lengts = [valid_periods[i,1] - valid_periods[i,0] for i in range(valid_periods.shape[0])]
    print(f'\nNumber of samples with less or equal than {less_than} values: {np.sum(np.where(np.array(period_lengts) <= less_than, 1, 0))}')
    print(f'Number of samples with more or equal than {higher_than} values: {np.sum(np.where(np.array(period_lengts) >= higher_than, 1, 0))}')
    print(f'Average number of values: {np.average(period_lengts):.5f}')
    print(f'Median number of values: {np.median(period_lengts):.5f}')