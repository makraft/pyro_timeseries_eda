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
file_path = found_files[0]['file path']



# %%
# open file as dataframe. This method is a lot faster (about 5x) than using 
# readline() to create lists and zip those into a dataframe.
import time
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
