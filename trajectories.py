import os
import numpy as np
from sklearn.model_selection import train_test_split

class TrajectoryDatasetProcessor:
    def __init__(self, obs_len=8, pred_len=12, test_size=0.2, random_state=42):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.test_size = test_size
        self.random_state = random_state

    def load_pedestrian_data(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data = []
        for line in lines:
            line = line.strip().split()
            coordinates = [float(x) for x in line]
            data.append(coordinates)

        return np.array(data)

    def split_pedestrians(self, dataset_path):
        try:
            file_paths = []
            for file_name in os.listdir(dataset_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(dataset_path, file_name)
                    file_paths.append(file_path)
        except Exception as e:
            print(f"Error listing files in {dataset_path}: {e}")
            return []

        all_data = []
        for file_path in file_paths:
            try:
                data = self.load_pedestrian_data(file_path)
                all_data.append(data)
            except Exception as e:
                print(f"Error loading data from {file_path}: {e}")

        return all_data

    def split_train_val_test(self, x, t):
        x_train, x_temp, y_train, y_temp = train_test_split(x, t, test_size=self.test_size, random_state=self.random_state)
        x_val, x_test, y_val, y_test_gt = train_test_split(x_temp, y_temp, test_size=0.5, random_state=self.random_state)

        return x_train, y_train, x_val, y_val, x_test, y_test_gt

    def process_data_array(self, data_array):
        x = []
        t = []

        for i in range(len(data_array) - self.obs_len - self.pred_len):
            obs_data = data_array[i:i + self.obs_len]
            pred_data = data_array[i + self.obs_len: i + self.obs_len + self.pred_len]

            # Reshape observation and prediction data
            obs_data = np.array(obs_data).reshape((self.obs_len, -1, 2))
            pred_data = np.array(pred_data).reshape((self.pred_len, -1, 2))

            x.append(obs_data)
            t.append(pred_data)

        return np.array(x), np.array(t)

    def process_all_data(self, dataset_path):
        x_train, y_train, x_val, y_val, x_test, y_test_gt = [], [], [], [], [], []

        for folder_name in ['train', 'val', 'test']:
            folder_path = os.path.join(dataset_path, folder_name)
            all_data = self.split_pedestrians(folder_path)

            if not all_data:
                print(f"No data found in {folder_name}.")
                continue

            x_data = []
            t_data = []

            for data_array in all_data:
                x_subset, t_subset = self.process_data_array(data_array)
                x_data.append(x_subset)
                t_data.append(t_subset)

            if not x_data:
                print(f"No processed data found in {folder_name}.")
                continue

            x_data = np.concatenate(x_data)
            t_data = np.concatenate(t_data)

            if folder_name == 'train':
                x_train, y_train = self.split_train_val_test(x_data, t_data)[:2]
            elif folder_name == 'val':
                x_val, y_val = self.split_train_val_test(x_data, t_data)[2:4]
            elif folder_name == 'test':
                x_test, y_test_gt = self.split_train_val_test(x_data, t_data)[4:]

        return x_train, y_train, x_val, y_val, x_test, y_test_gt

# Example usage:
dataset_path = '/Users/hanse/Documents/Research/datasets/eth/'
obs_len = 8
pred_len = 12

dataset_processor = TrajectoryDatasetProcessor(obs_len=obs_len, pred_len=pred_len)
x_train, y_train, x_val, y_val, x_test, y_test_gt = dataset_processor.process_all_data(dataset_path)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)
print("x_test shape:", x_test.shape)
print("y_test_gt shape:", y_test_gt.shape)
