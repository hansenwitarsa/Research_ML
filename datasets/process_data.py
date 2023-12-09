from simplified_frame import SimplifiedFrame
from sort_id import SortPedestrianID
from split_train_val import SplitDataByPedestrianID

# Simplified Frame
input_file = "/Users/hanse/Documents/Research/datasets_process/raw/all_data/zara03_posvel.txt"
output_file = "/Users/hanse/Documents/Research/datasets_process/zara03/test/zara03.txt"

frame_mapper = SimplifiedFrame(input_file, output_file)
frame_mapper.process_data()

# Sort Pedestrian ID
input_file_2 = "/Users/hanse/Documents/Research/datasets_process/zara03/test/zara03.txt"
output_file_2= "/Users/hanse/Documents/Research/datasets_process/zara03/test/zara03.txt"

pedestrian_sorter = SortPedestrianID(input_file_2, output_file_2)
pedestrian_sorter.process_data()

# Split Data by Pedestrian ID
#---------- split id:
# eth: 256, hotel: 320, uni: 89, zara01: 110, zara02: 144, zara03: 128
input_file_3 = "/Users/hanse/Documents/Research/datasets_process/zara03/test/zara03.txt"
output_file_below_split_id = "/Users/hanse/Documents/Research/datasets_process/raw/train/zara03_train.txt"
output_file_above_split_id = "/Users/hanse/Documents/Research/datasets_process/raw/val/zara03_val.txt"

splitter = SplitDataByPedestrianID(input_file_3, output_file_below_split_id, output_file_above_split_id, split_id = 128)
splitter.process_data()


# This class is to convert a certain column into integer
# class ConvertSecondColumnToInteger:
#     def __init__(self, input_file, output_file):
#         self.input_file = input_file
#         self.output_file = output_file

#     def process_data(self):
#         # Read the data from the input file
#         with open(self.input_file, 'r') as file:
#             data = file.readlines()

#         # Convert the values in the second column to integers
#         processed_data = []
#         for line in data:
#             columns = line.split()
#             if len(columns) >= 2:  # Ensure the line has at least 2 columns
#                 # Convert the value in the second column to an integer
#                 columns[1] = str(int(float(columns[1])))
#                 processed_data.append(' '.join(columns) + '\n')

#         # Write the processed data to a new file
#         with open(self.output_file, 'w') as output_file:
#             output_file.writelines(processed_data)

# # Example usage:
# input_file = "/Users/hanse/Documents/Research/datasets_process/hotel/test/biwi_hotel.txt"
# output_file = "/Users/hanse/Documents/Research/datasets_process/hotel/test/biwi_hotel.txt"
# column_converter = ConvertSecondColumnToInteger(input_file, output_file)
# column_converter.process_data()
