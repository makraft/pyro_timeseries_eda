# %%    Find data directory
# define directory and test if path works
import os

# Define the path to your data file
data_path = "D:\\mkraft\\layer_thickness_build_1"

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
layer_thickness = "30" # Specify the desired layer thickness

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
plt.close()

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

plt.plot(df_data['t'] / 1e6, df_data['intensity'], label='intensity')
plt.plot(df_data['t'] / 1e6, df_data['status']*100, 'r', label='state * 100')
plt.plot(filtered_df['t'] / 1e6, filtered_df['intensity'], 'g', label='intensity filtered')
plt.xlabel("Timestamp [s]")
plt.ylabel("Intensity [mV]")
plt.title('Intensity profile over time')
plt.grid(axis='y')
plt.legend()
plt.show()


# %% FFT plots
# Generate plots of FFT data of the individual scan vectors

import numpy as np

# Identify the window boundaries where the mask changes from True to False and vice versa
start_indices = mask[(mask.shift() == False) & (mask == True)].index
end_indices = mask[(mask == True) & (mask.shift() == False)].index

# Set the window size for the FFT
window_size = 1024

# Perform FFT on each window and plot the frequency spectrum
""" for i in range(len(window_boundaries) - 1):
    start_index = window_boundaries[i]
    end_index = window_boundaries[i + 1]
    
    window_data = filtered_df['y'].values[start_index:end_index]
    window_time = filtered_df['t'].values[start_index:end_index]
    
    # Apply FFT to the window data
    fft_result = np.fft.fft(window_data)
    frequencies = np.fft.fftfreq(len(window_data), window_time[1] - window_time[0])
    
    # Plot the frequency spectrum
    plt.figure()
    plt.plot(frequencies, np.abs(fft_result))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum - Window {}'.format(i+1))
    plt.show()
 """