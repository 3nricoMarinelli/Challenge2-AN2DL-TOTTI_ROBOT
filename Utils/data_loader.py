import numpy as np
import pandas as pd

def train_valid_test_split(training_data, valid_periods, categories, train_ratio, valid_ratio):
  columns = ["training", "valid", "test", "category"]
  data = pd.DataFrame(index=range(training_data.shape[0]), columns=columns)

  for i in range(training_data.shape[0]):
    start = valid_periods[i,0]
    end = valid_periods[i,1]
    non_zero_samples = end - start
    train_len = round(non_zero_samples*train_ratio)
    valid_len = round(non_zero_samples*valid_ratio)
    test_len = non_zero_samples-train_len-valid_len

    training_samples=training_data[i, start : start + train_len]
    valid_samples=training_data[i, start+train_len : start+train_len+valid_len]
    test_samples=training_data[i,start+train_len+valid_len:]

    data.loc[i] = [training_samples, valid_samples, test_samples, categories[i]]
  return data

def build_sequences(df, window=200, stride=20, telescope=100):
    # Sanity check to avoid runtime errors
    assert window % stride == 0
    dataset = []
    labels = []
    temp_df = df.copy()
    padding_check = df.size%window

    if(padding_check != 0):
        # Compute padding length
        padding_len = window - df.size%window
        padding = np.zeros((padding_len), dtype='float32')
        temp_df = np.concatenate((padding,df))
        assert temp_df.size % window == 0

    #print(temp_df.size)
    for idx in np.arange(0,temp_df.size-window-telescope,stride):
        dataset.append(temp_df[idx:idx+window])
        labels.append(temp_df[idx+window:idx+window+telescope])

    if len(dataset) == 0:
      return dataset, labels
    else:
      return np.vstack(dataset), np.vstack(labels)
    
#data is a pandas series containing 48000 lists (either training, validation or test)
def build_sequence_dataset(data):
  dataset = []
  labels = []
  for i in range(data.size):
    dset, labs = build_sequences(data[i], window=10, stride=5, telescope=1)
    if len(dset) == 0:
      continue
    dataset.append(dset)
    labels.append(labs)

  return np.array(dataset,dtype=object), np.array(labels,dtype=object)

def THE_SEQUENCER(dataset):
  training_dataset, training_labels = build_sequence_dataset(dataset["training"])
  valid_dataset, valid_labels = build_sequence_dataset(dataset["valid"])
  test_dataset, test_labels = build_sequence_dataset(dataset["test"])
  return training_dataset, training_labels, valid_dataset, valid_labels, test_dataset, test_labels