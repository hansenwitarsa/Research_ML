class SplitDataByPedestrianID:
    def __init__(self, input_file, output_file_1, output_file_2, split_id):
        self.input_file = input_file
        self.output_file_1 = output_file_1
        self.output_file_2 = output_file_2
        self.split_id = split_id

    def process_data(self):
        # Read the data from the input file
        with open(self.input_file, 'r') as file:
            data = file.readlines()

        # Split the data based on the pedestrian ID
        data_below_split_id = [line for line in data if float(line.split()[1]) <= self.split_id]
        data_above_split_id = [line for line in data if float(line.split()[1]) > self.split_id]

        # Write the split data to separate files
        with open(self.output_file_1, 'w') as file_1:
            file_1.writelines(data_below_split_id)

        with open(self.output_file_2, 'w') as file_2:
            file_2.writelines(data_above_split_id)
