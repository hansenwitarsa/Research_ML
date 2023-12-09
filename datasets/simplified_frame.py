class SimplifiedFrame:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.frame_mapping = {}
        self.new_frame_number = 1

    def process_data(self):
        # Read the input data
        with open(self.input_file, "r") as file:
            lines = file.readlines()

        # Process each line in the dataset
        output_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 1:
                original_frame_number = float(parts[0])
                if original_frame_number not in self.frame_mapping:
                    self.frame_mapping[original_frame_number] = self.new_frame_number
                    self.new_frame_number += 1
                new_frame = self.frame_mapping[original_frame_number]
                modified_line = f"{new_frame} {' '.join(parts[1:])}"
                output_lines.append(modified_line)

        # Write the modified data to a new file
        with open(self.output_file, "w") as file:
            file.writelines("\n".join(output_lines))

