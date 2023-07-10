# %%    Find data directory
# define directory and test if path works
import os

# Define the path to your data file
#data_path = "D:\\mkraft\\layer_thickness_build_1"
data_path = "C:\\Users\\msskr\\Documents\\Master_Thesis\\Data\\layer_thickness_build_1"

# Check if the file exists
if os.path.isdir(data_path):
    print("Directory found!")
else:
    print("Directory not found. Please check the file path.")




# %%    Find files
# Create set of paths to access individual files
import pandas as pd

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
df = pd.DataFrame(file_attributes, columns=["file path", "part", "iteration", "layer thickness"])



# %%    Select single file
# Find the path to a selected file
import matplotlib.pyplot as plt

# Define the criteria for selecting the file
part = "3"  # Specify the desired part number
iteration = "5"  # Specify the desired iteration of the layer experiment
layer_thickness = "20" # Specify the desired layer thickness

# Filter the DataFrame based on the criteria
selected_files = df[(df['part'] == part) & (df['iteration'] == iteration) & (df['layer thickness'] == layer_thickness)]

# Iterate over the selected files
#for _, file_row in selected_files.iterrows():
found_files=[]
for _, file_row in selected_files.iterrows():
    found_files.append(file_row)
# Get the file path from the DataFrame
file_path = found_files[0]['file path']



# %%    Dataframe from single file
# open file as dataframe
import time
import numpy as np
tstart = time.time()
cols=columns=['t','x', 'y','z','intensity','sensor1','sensor2','sensor3','status','controller']
df_data=pd.read_csv(file_path, delimiter=" ", skiprows=26,dtype=np.int32,names=cols)
print("Time for importing data with pd.read_csv: {0}".format((str(time.time()-tstart))))



# %%    Scatterplots for individual time series

# Generate single part scatter plot
# Filter the DataFrame based on the "status" column
filtered_df = df_data[df_data['status'] == 1]
# If desired, set limits to the colorbar. Important for comparing plots.
set_uniform_limits = True
max_intensity = df_data['intensity'].max()
if set_uniform_limits:
    colorbar_min = 0    # custom lower limit
    colorbar_max = 155  # custom upper limit
else:
    # The colorbar is adjusted automatically for each plot
    colorbar_min = df_data['intensity'].min()
    colorbar_max = max_intensity
if max_intensity > colorbar_max:
    print("Warning, some large intensity values are not plotted correctly!")
    # Adjust colorbar_max if this occurs
# Create the scatterplot
plt.scatter(filtered_df["x"],filtered_df["y"], c=filtered_df["intensity"], cmap="hot",s=1, vmin=colorbar_min, vmax=colorbar_max)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Layer thickness {0} [micro_m], Part {1}, Iteration {2}'.format(layer_thickness, part, iteration))
plt.colorbar(label='Intensity, Maximum = {0}'.format(max_intensity))
#plt.show()
#plt.close()

# %%
# Create value over time plot

# Identify the indices where 'Status' switches from 0 to 1
status_switch_indices = df_data[(df_data['status'].shift() == 0) & (df_data['status'] == 1)].index

# Create a boolean mask to exclude the next 30 points after each status switch
# Also exclude all 0 status points
mask = pd.Series(True, index=df_data.index)
mask[df_data['status'] == 0] = False
for index in status_switch_indices:
    mask[index : index + 31] = False

# Apply the boolean mask to filter the data
filtered_df = df_data[mask]

plt.clf()
plt.plot(df_data['t'] / 1e6, df_data['intensity'], label='intensity')
plt.plot(df_data['t'] / 1e6, df_data['status']*100, 'r', label='state * 100')
plt.plot(filtered_df['t'] / 1e6, filtered_df['intensity'], 'g', label='intensity filtered')
plt.xlabel("Timestamp [s]")
plt.ylabel("Intensity [mV]")
plt.title('Intensity profile over time')
plt.grid(axis='y')
plt.legend()
plt.show()
plt.close()


# %% FFT plots
# Generate plots of FFT data of the individual scan vectors

import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.colors as mcolors

sampling_rate=100000

# Identify the window boundaries where the mask changes from True to False and vice versa
start_indices = mask[(mask.shift() == False) & (mask)].index
end_indices = mask[(mask.shift() == True) & (~mask)].index


plt.clf()
# Create a figure for the plots
plt.figure()
color_names= list(mcolors.CSS4_COLORS.keys())
# Iterate over start and end indices
for i, (start, end) in enumerate(zip(start_indices, end_indices)):
    # Select data within the current interval
    interval_data = df_data.loc[start:end, 'intensity']

    # Calculate the FFT of the intensity data in the current interval
    yf = fft(interval_data)
    xf = fftfreq(len(interval_data),1.0/sampling_rate)

    # Determine the color for this line
    color = color_names[i % len(color_names)]

    # Plot the absolute value of the FFT result, only positive side and without DC component (xf = 0)
    plt.plot(xf[1:len(xf)//2], np.abs(yf[1:len(yf)//2]), color=color, label="vector {0}".format(i))

    # Set the plot labels
    plt.xlabel('Frequency [1/s]')
    plt.ylabel('Magnitude')

#    if i > 5:
#        break
# Show the plot
plt.tight_layout()
plt.legend()
plt.show()



# %% Mean values evolution
# Calculate and plot the mean values for each interval
import numpy as np

# Identify the window boundaries where the mask changes from True to False and vice versa
start_indices = mask[(mask.shift() == False) & (mask)].index
end_indices = mask[(mask.shift() == True) & (~mask)].index

# Initialize an empty list to store the mean values
mean_values = []

# Iterate over start and end indices
for start, end in zip(start_indices, end_indices):
    # Select data within the current interval
    interval_data = df_data.loc[start:end, 'intensity']

    # Calculate the mean value of the interval data and append to the list
    mean_values.append(interval_data.mean())

# Create a figure for the plot
plt.figure(figsize=(12, 8))

# Plot the mean values
plt.plot(mean_values)

# Set the plot labels
plt.xlabel('Vector')
plt.ylabel('Mean Intensity')

# Show the plot
plt.tight_layout()
plt.show()


# %% Plot multiple means
# Plot evolution of mean value over multiple files

import numpy as np
# Define the criteria for selecting the files
part = "3"  # Specify the desired part number
iteration = "5"  # Specify the desired iteration of the layer experiment

# Filter the DataFrame based on the criteria
selected_files = df[(df['part'] == part) & (df['iteration'] == iteration)]

# Group selected files by layer thickness
grouped_files = selected_files.groupby('layer thickness')

# Initialize empty lists to store mean values for each layer thickness
mean_values_dict = {}

# Iterate over each group
for layer_thickness, group in grouped_files:
    mean_values = []
    # Iterate over each file in the group
    for _, file_row in group.iterrows():
        # Get the file path from the DataFrame
        file_path = file_row['file path']

        # Load the data into a DataFrame
        cols = ['t', 'x', 'y', 'z', 'intensity', 'sensor1', 'sensor2', 'sensor3', 'status', 'controller']
        df_data = pd.read_csv(file_path, delimiter=" ", skiprows=26, dtype=np.int32, names=cols)

        # Create a boolean mask to exclude the next 30 points after each status switch
        # Also exclude all 0 status points
        mask = pd.Series(True, index=df_data.index)
        mask[df_data['status'] == 0] = False
        for index in status_switch_indices:
            mask[index : index + 31] = False
        
        # Identify the window boundaries where the mask changes from True to False and vice versa
        start_indices = mask[(mask.shift() == False) & (mask)].index
        end_indices = mask[(mask.shift() == True) & (~mask)].index

        # Iterate over start and end indices
        for start, end in zip(start_indices, end_indices):
            # Select data within the current interval
            interval_data = df_data.loc[start:end, 'intensity']

            # Calculate the mean value of the interval data and append to the list
            mean_values.append(interval_data.mean())

    # Store mean values for this layer thickness
    mean_values_dict[layer_thickness] = mean_values

# Create a figure for the plot
plt.figure(figsize=(12, 8))

# Plot the mean values
for key, values in mean_values_dict.items():
    plt.plot(values,label="{0}".format(key))

# Set the plot labels
plt.xlabel('Vector')
plt.ylabel('Mean Intensity')

# Show the plot
plt.legend(title="Layer thickness [microns]")
plt.tight_layout()
plt.show()


# %% Plot means to compare between layers
# Create violon and box plot for one part, one iteration with all layer thicknesses

import numpy as np
# Define the criteria for selecting the files
part = "3"  # Specify the desired part number
iteration = "5"  # Specify the desired iteration of the layer experiment

# Filter the DataFrame based on the criteria
selected_files = df[(df['part'] == part) & (df['iteration'] == iteration)]

# Group selected files by layer thickness
grouped_files = selected_files.groupby('layer thickness')

# Initialize empty lists to store mean values for each layer thickness
mean_values_dict = {}

# Iterate over each group
for layer_thickness, group in grouped_files:
    mean_values = []
    # Iterate over each file in the group
    for _, file_row in group.iterrows():
        # Get the file path from the DataFrame
        file_path = file_row['file path']

        # Load the data into a DataFrame
        cols = ['t', 'x', 'y', 'z', 'intensity', 'sensor1', 'sensor2', 'sensor3', 'status', 'controller']
        df_data = pd.read_csv(file_path, delimiter=" ", skiprows=26, dtype=np.int32, names=cols)

        # Create a boolean mask to exclude the next 30 points after each status switch
        # Also exclude all 0 status points
        mask = pd.Series(True, index=df_data.index)
        mask[df_data['status'] == 0] = False
        for index in status_switch_indices:
            mask[index : index + 31] = False
        
        # Identify the window boundaries where the mask changes from True to False and vice versa
        start_indices = mask[(mask.shift() == False) & (mask)].index
        end_indices = mask[(mask.shift() == True) & (~mask)].index

        # Iterate over start and end indices
        for start, end in zip(start_indices, end_indices):
            # Select data within the current interval
            interval_data = df_data.loc[start:end, 'intensity']

            # Calculate the mean value of the interval data and append to the list
            mean_values.append(interval_data.mean())

    # Store mean values for this layer thickness
    mean_values_dict[layer_thickness] = mean_values

# Create a figure for the plot
all_data=list(mean_values_dict.values())
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
axs[0].violinplot(all_data,
                  showmeans=False,
                  showmedians=True)
axs[0].set_title('Violin plot')

# plot box plot
axs[1].boxplot(all_data)
axs[1].set_title('Box plot')

# adding horizontal grid lines
for ax in axs:
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(all_data))],
                  labels=['0', '10', '20', '30', '40', '50','60','70','80','90'])
    ax.set_xlabel('Layer thickness [microns]')
    ax.set_ylabel('Mean Intensity [mV]')

plt.show()
