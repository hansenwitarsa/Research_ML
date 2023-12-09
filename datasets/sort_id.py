class SortPedestrianID:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def process_data(self):
        # Read the data from the text file
        with open(self.input_file, 'r') as file:
            data = file.readlines()

        # Split each line into (frame, pedestrian_id, x, y) components
        data = [line.split() for line in data]

        # Sort the data by pedestrian ID (the second column)
        sorted_data = sorted(data, key=lambda x: float(x[1]))

        # Reconstruct the lines with sorted data
        sorted_lines = [' '.join(row) + '\n' for row in sorted_data]

        # Write the sorted data to a new file
        with open(self.output_file, 'w') as output_file:
            output_file.writelines(sorted_lines)

