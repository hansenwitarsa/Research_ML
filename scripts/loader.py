import numpy as np
import os
import sys
import tensorflow as tf

# Create 3D array
class DataIntoArray:
    def __init__(self, path):
        self.path = path # File path
        self.data = np.loadtxt(path) # Load ata inside the path
        self.totalnum_ped = int(np.max(self.data[:,1])) -  int(np.min(self.data[:,1])) + 1 # Find the total number of pedestrian in the data through the id

    def total_frame(self, datapart): # Find out total number of frame
        max_value = -float('inf')  # Initialize with negative infinity
        min_value = float('inf')   # Initialize with positive infinity

        for entry in datapart:
            frame_number = int(entry[0])
            max_value = max(max_value, frame_number)
            min_value = min(min_value, frame_number)

        return max_value - min_value + 1

    def process_data(self, obs_len = 8, pred_len = 12, max_ped = 64):

        # Turn all frame and id sos that they starts with 1
        min_frame_all = min(self.data[:, 0])
        min_id_all = min(self.data[:, 1])
        self.data[:, 0] = self.data[:, 0] - min_frame_all + 1
        self.data[:, 1] = self.data[:, 1] - min_id_all + 1
        
        num_parts = ((self.totalnum_ped - 1) // (max_ped)) + 1 # Find out how many parts if we divide by 64
        parts = [[] for _ in range(num_parts)] # Initiliaze array that will contain np.zeros for each part
        x_data = [] # Obs data
        t_data = [] # Pred data
 
        for part_id in range(num_parts):
            start_id = (part_id * max_ped) + 1 # For each part, start with pedestrian with id berapa?
            end_id = min((part_id + 1) * max_ped, self.totalnum_ped) # And end with pedestrian with id berapa?

            part_data = self.data[(self.data[:, 1] >= start_id) & (self.data[:, 1] <= end_id)] # Input from the initial dataset to part data
            
            min_frame = min(part_data[:, 0]) # Find minimum value of frame
            min_ped_id = min(part_data[:, 1]) # Find minimum value of pedestrian id

            parts[part_id] = (np.zeros((self.total_frame(part_data), 64, 2), dtype=float)) # For each parts[i], fill with 3d np.zeros that later will be filled with x and y coor

            for entry in part_data:
                frame_number = int(int(entry[0]) - min_frame)  # Convert to 0-based index, nanti dimasukin ke partsnya gampang
                pedestrian_id = int(int(entry[1]) - min_ped_id)  # Convert to 0-based index
                x, y = entry[2], entry[3]

                # Fill the parts array with the x and y coordinates
                parts[part_id][frame_number][pedestrian_id][0] = x
                parts[part_id][frame_number][pedestrian_id][1] = y
        
            for i in range(len(parts[part_id]) - obs_len - pred_len):
                # Append a chunk of data with shape (obs_len, 64, 2) to x_data
                x_data.append(parts[part_id][i:i + obs_len])
                # Append a chunk of data with shape (pred_len, 64, 2) to t_data
                t_data.append(parts[part_id][i + obs_len: i + obs_len + pred_len])

        return x_data, t_data
    
    def process_folder(folder_path, obs_len=8, pred_len=12, max_ped=64):
        # Get a list of subfolder files in the folder
        subfolders = ['train', 'val', 'test']

        # To store the joined data
        x_train, t_train = [], []
        x_val, t_val = [], []
        x_test, t_test = [], []

        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)

            dataset_files = [f for f in os.listdir(subfolder_path) if f.endswith('.txt')]

            if subfolder == 'train':
                for dataset_file in dataset_files:
                    dataset_path = os.path.join(subfolder_path, dataset_file)
                
                    # Create an instance of the class for each dataset
                    data_processor = DataIntoArray(dataset_path)

                    # Process the data for the current dataset
                    x_data, t_data = data_processor.process_data(obs_len, pred_len, max_ped)
          
                    #add into all data
                    x_train += x_data
                    t_train += t_data

            if subfolder == 'val':
                for dataset_file in dataset_files:
                    dataset_path = os.path.join(subfolder_path, dataset_file)
                
                    # Create an instance of the class for each dataset
                    data_processor = DataIntoArray(dataset_path)

                    # Process the data for the current dataset
                    x_data, t_data = data_processor.process_data(obs_len, pred_len, max_ped)
          
                    #add into all data
                    x_val += x_data
                    t_val += t_data

            if subfolder == 'test':
                for dataset_file in dataset_files:
                    dataset_path = os.path.join(subfolder_path, dataset_file)
                
                    # Create an instance of the class for each dataset
                    data_processor = DataIntoArray(dataset_path)

                    # Process the data for the current dataset
                    x_data, t_data = data_processor.process_data(obs_len, pred_len, max_ped)
          
                    #add into all data
                    x_test += x_data
                    t_test += t_data

        # Turn list into array
        x_train = np.array(x_train)
        t_train = np.array(t_train)
        x_val = np.array(x_val)
        t_val = np.array(t_val)
        x_test = np.array(x_test)
        t_test = np.array(t_test)

        # train = tf.concat([x_train, t_train], axis=1)
        # val = tf.concat([x_val, t_val], axis=1)
        # test = tf.concat([x_test, t_test], axis=1)

        return x_train, t_train, x_val, t_val, x_test, t_test

# folder_path = "/Users/hanse/Documents/Research/datasets/eth/"

# x_train, y_train, x_val, y_val, x_test, y_test_gt = DataIntoArray.process_folder(folder_path, obs_len = 8, pred_len = 12, max_ped = 64)
# print("x_train shape:", x_train.shape)
# print("y_train shape:", y_train.shape)
# print("x_val shape:", x_val.shape)
# print("y_val shape:", y_val.shape)
# print("x_test shape:", x_test.shape)
# print("y_test shape:", y_test_gt.shape)
# print(y_test_gt[0])
# print(x_test[0])

# def save_to_file(file_path, data):
#     with open(file_path, 'w') as f:
#         # Redirect standard output to the file
#         sys.stdout = f

#         # Print the content of the data
#         print(data)

#         # Reset standard output
#         sys.stdout = sys.__stdout__
# file_path = "/Users/hanse/Documents/Research/scripts/example.txt"
# ex = x_test[0]
# save_to_file(file_path=file_path, data=ex)
