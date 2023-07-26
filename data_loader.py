# %% Definitions
# Define custon dataset and dataloader

import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, sequence_length, batch_size=1, shuffle=False, num_workers=0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.sequence_length = sequence_length
        self.shuffle=shuffle

    def __iter__(self):
        return _CustomDataLoaderIter(self)

class _CustomDataLoaderIter:
    # Do custom iteration of the data.
    # todo: figure out if using a sampler is better: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
    # maybe also consider this thread: https://discuss.pytorch.org/t/overwriting-pytorch-dataloader-shuffle/146273
    def __init__(self, loader):
        self.loader = loader
        self.indices = torch.arange(len(loader.dataset))
        if loader.shuffle:
            self.indices = self.indices[torch.randperm(len(self.indices))]
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.indices):
            raise StopIteration
        data_idx = self.indices[self.idx]
        data, label = self.loader.dataset[data_idx]
        if len(data) >= self.loader.sequence_length:
            start_index = torch.randint(0, len(data) - self.loader.sequence_length + 1, (1,))
            end_index = start_index + self.loader.sequence_length
            self.idx += 1
            return data[start_index:end_index], label
        else:
            self.idx += 1
            return next(self)



# Define the path to your data file
data_path = "D:\\mkraft\\layer_thickness_build_1"   # AMLZ Lab PC
#data_path = "C:\\Users\\msskr\\Documents\\Master_Thesis\\Data\\layer_thickness_build_1"    # Home PC

# Create dataset
tstart = time.time()
dataset = PyroTimeSeriesDataset(data_path)
tend=time.time()
print("Time for initializing the dataset: {0} seconds".format(tend-tstart))
print("Length of dataset: {0} sequences".format(len(dataset)))

# Create dataloader
tstart = time.time()
sequence_length=10
custom_dataloader=CustomDataLoader(dataset, sequence_length, batch_size=1, shuffle=False, num_workers=0)
tend=time.time()
print("Time for initializing the dataloader: {0} seconds".format(tend-tstart))
i = 0
for data, label in custom_dataloader:
    print(data, label)
    i+=1
    if i>20:
        break