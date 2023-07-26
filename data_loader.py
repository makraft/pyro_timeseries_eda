# %% Definitions
# Define custon dataset and dataloader

import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import random_split
import pandas as pd
import numpy as np
import time

class PyroTimeSeriesDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data=[]
        self.labels=[]

        # Check if the file exists
        if os.path.isdir(data_path):
            print("Directory found!")
        else:
            print("Directory not found. Please check the file path.")


        # Create set of paths to access individual files

        # Initialize an empty list to store file attributes
        file_attributes = []

        # Recursive function to extract file attributes from subdirectories
        def extract_attributes(directory_path):
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if file.endswith(".pcd"):
                        # Get the file path
                        file_path = os.path.join(root, file)
                        # Extract attributes based on subfolder
                        split_path=file_path.split(os.sep)
                        part = split_path[-2]
                        iteration = split_path[-4]
                        layer_thickness = split_path[-5].split("_")[0]
                        file_attributes.append((file_path, part, iteration, layer_thickness, ))

        # Call the function with the parent directory path
        extract_attributes(data_path)

        # Create a DataFrame from the list of file attributes
        df_files = pd.DataFrame(file_attributes, columns=["file path", "part", "iteration", "layer thickness"])

        # Optional: filter files to only include certain data
        # Define the criteria for selecting the files
        part = "3"  # Specify the desired part number
        iteration = "5"  # Specify the desired iteration of the layer experiment

        # Filter the DataFrame based on the criteria
        selected_files = df_files[(df_files['part'] == part) & (df_files['iteration'] == iteration)]

        # Standard: do not filter dataset
#        selected_files = df_files

        # Group selected files by layer thickness
        grouped_files = selected_files.groupby('layer thickness')

        # Iterate over each group
        for layer_thickness, group in grouped_files:
            # Iterate over each file in the group
            for _, file_row in group.iterrows():
                # Get the file path from the DataFrame
                file_path = file_row['file path']

                # Load the data into a DataFrame
                cols = ['t', 'x', 'y', 'z', 'intensity', 'sensor1', 'sensor2', 'sensor3', 'status', 'controller']
                df_data = pd.read_csv(file_path, delimiter=" ", skiprows=26, dtype=np.int32, names=cols)

                # Create a boolean mask to exclude points
                # Identify the indices where 'Status' switches from 0 to 1
                status_switch_indices = df_data[(df_data['status'].shift() == 0) & (df_data['status'] == 1)].index
                mask = pd.Series(True, index=df_data.index)
                # Exclude all 0 status points
                mask[df_data['status'] == 0] = False
                # Exclude the next 30 points after each status switch
                for index in status_switch_indices:
                    mask[index : index + 31] = False
                
                # Identify the window boundaries where the mask changes from True to False and vice versa
                start_indices = mask[(mask.shift() == False) & (mask)].index
                end_indices = mask[(mask.shift() == True) & (~mask)].index

                # Iterate over start and end indices
                for start, end in zip(start_indices, end_indices):
                    # Select data within the current interval
                    interval_data = df_data.loc[start:end, 'intensity']
                    self.data.append(interval_data.values)
                    self.labels.append(layer_thickness)
                print("Number of vectors with {0} layer thickness: {1}",format([layer_thickness, len(start_indices)]))
            
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, index):
        data=self.data[index]
        label=self.labels[index]
        return(torch.Tensor(data), label)


class SequenceSampler(Sampler):
    """
    Custom sampler in pytorch style. Also works without inheritance.
    """
    def __init__(self, dataset, min_sequence_length, max_sequence_length):
        self.dataset = dataset
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.indices = self.generate_indices()

    def generate_indices(self):
        # note that idx does not consider the self.dataset.indices list
        indices = []
        for idx, data in enumerate(self.dataset):
            sequence_length = len(data[0])
            num_samples = (sequence_length // self.max_sequence_length) + 1
            for _ in range(num_samples):
                # todo: check if proper range for randint
                sample_length = torch.randint(self.min_sequence_length, self.max_sequence_length + 1, (1,)).item()
                start_index = torch.randint(0, sequence_length - sample_length + 1, (1,)).item()
                indices.append((idx, start_index, sample_length))
        return indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

class SampledDataset(Dataset):
    def __init__(self, base_dataset, sampler):
        self.base_dataset = base_dataset
        self.sampler = sampler

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        idx, start_index, sample_length = self.sampler.indices[index]
        data = self.base_dataset[idx][0][start_index : start_index + sample_length]
        label = self.base_dataset[idx][1]
        return data, label





# Define the path to your data file
data_path = "D:\\mkraft\\layer_thickness_build_1"   # AMLZ Lab PC
#data_path = "C:\\Users\\msskr\\Documents\\Master_Thesis\\Data\\layer_thickness_build_1"    # Home PC

# Create dataset
tstart = time.time()
dataset = PyroTimeSeriesDataset(data_path)
tend=time.time()
print("Time for initializing the dataset: {0} seconds".format(tend-tstart))
print("Length of dataset: {0} sequences".format(len(dataset)))

# Create a Test/Train split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# Generate new dataset by sampling sequences
tstart = time.time()
min_sequence_length = 50
max_sequence_length = 1000
sampler = SequenceSampler(train_dataset, min_sequence_length, max_sequence_length)
train_dataset_sampled = SampledDataset(train_dataset,sampler)
tend=time.time()
print("Time for creating the sampled dataset: {0} seconds".format(tend-tstart))
print("Total number of samples: {0}".format(len(train_dataset_sampled)))

# Create dataloader
tstart = time.time()
train_dataloader = DataLoader(train_dataset_sampled)
tend=time.time()
print("Time for initializing the dataloader: {0} seconds".format(tend-tstart))

# %%
# Analyze samples
import matplotlib.pyplot as plt
from collections import Counter

sample_lengths = []
thicknesses = []
i = 0
for data, label in train_dataloader:
    length=len(data[0])
    i+=1
    if i<20:
        print("Sample length: {0}, label: {1}".format(len(data[0]), label))
    sample_lengths.append(length)
    thicknesses.append(int(label[0]))

# Plot distribution of sample lengths
plt.hist(x=sample_lengths,bins=19, edgecolor="black")
plt.grid(True)
plt.title("Distribution of sample lengths")
plt.xlabel("Sample length")
plt.ylabel("Number of samples")
plt.show()

# Plot distribution of layer thicknesses in samples
plt.grid(True)
#plt.hist(x=thicknesses, edgecolor="black")
plt.title("Distribution of sample thicknesses")
plt.ylabel("Number of samples")
plt.xlabel("Layer thickness")
# Count the occurrences of each unique value in the list
value_counts = Counter(thicknesses)
# Extract the unique values and their corresponding counts
values, counts = zip(*value_counts.items())
plt.bar(values, counts, width=8, edgecolor="black")
plt.show()
