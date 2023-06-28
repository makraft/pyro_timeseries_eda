# %%
# define directory and test if path works
import os

# Define the path to your data file
data_path = "D:\\mkraft\\layer_thickness_build_1"

# Check if the file exists
if os.path.isdir(data_path):
    print("Directory found!")
else:
    print("Directory not found. Please check the file path.")




# %%
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



# %%
# Find the path to a selected file
import matplotlib.pyplot as plt

# Define the criteria for selecting the file
part = "1"  # Specify the desired part number
iteration = "1"  # Specify the desired iteration of the layer experiment
layer_thickness = "0" # Specify the desired layer thickness

# Filter the DataFrame based on the criteria
selected_files = df[(df['part'] == part) & (df['iteration'] == iteration) & (df['layer thickness'] == layer_thickness)]

# Iterate over the selected files
#for _, file_row in selected_files.iterrows():
found_files=[]
for _, file_row in selected_files.iterrows():
    found_files.append(file_row)
# Get the file path from the DataFrame
file_path = file_row['file path']

# %%
# Initialize empty lists to store the extracted data
import time
tstart = time.time()
times = []
x_coords = []
y_coords = []
z_coords = []
intensities= []
laser_status=[]

# Open the file and read its contents
with open(file_path, 'r') as f:
    lines = f.readlines()
    # Find the line that says "DATA ascii" to identify the start of the data
    start_index = lines.index('DATA ascii\n') + 1

    # Read the subsequent lines and extract the required information
    for line in lines[start_index:]:
        # Split the line by spaces
        data = line.split()

        # Extract the required information from the columns
        t =int(data[0])
        x =int(data[1])
        y =int(data[2])
        z =int(data[3])
        intensity =int(data[4])
        status = int(data[8])

        # Append the extracted data to the respective lists
        times.append(t)
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        intensities.append(intensity)
        laser_status.append(status)

# Create a DataFrame from the data
    df_data = pd.DataFrame(
        list(zip(times,x_coords,y_coords,z_coords,intensities,laser_status)),
        columns=['t','x', 'y','z','intensity','status'])
print("Time for importing data with single array: {0}".format((str(time.time()-tstart))))



# %%
# Initialize empty lists to store the extracted data
import time
tstart = time.time()
data_list = []
# Open the file and read its contents
with open(file_path, 'r') as f:
    lines = f.readlines()
    # Find the line that says "DATA ascii" to identify the start of the data
    start_index = lines.index('DATA ascii\n') + 1

    # Read the subsequent lines and extract the required information
    for line in lines[start_index:]:
        # Split the line by spaces
        data = line.split()
        data_list.append(data)
    # Create a DataFrame from the data
    df_data = pd.DataFrame(data_list, columns=['t','x', 'y','z','intensity','sensor1','sensor2','sensor3','status','controller'])
print("Time for importing data with arrays: {0}".format((str(time.time()-tstart))))


# %%
import numpy as np
tstart = time.time()
cols=columns=['t','x', 'y','z','intensity','sensor1','sensor2','sensor3','status','controller']
df_data=pd.read_csv(file_path, delimiter=" ", skiprows=26,dtype=np.int32,names=cols)
print("Time for importing data with pd.read_csv: {0}".format((str(time.time()-tstart))))



# %%

# Create the scatterplot
plt.scatter(x_coords, y_coords, c=intensities, cmap="hot")
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Scatterplot')
plt.colorbar(label='Intensity')
plt.show()

# %%
