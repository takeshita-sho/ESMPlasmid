#/fs/cbcb-lab/mpop/projects/premature_microbiome/assembly/
# for all files ending in .fa
import os
import pickle

# Function to read a .fa file and create a list of non-header lines
def process_fa_file(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        current_sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = ''
            else:
                current_sequence += line
        if current_sequence:
            sequences.append(current_sequence)
    return sequences

# Directory where your .fa files are located
read_directory = '/fs/cbcb-lab/mpop/projects/premature_microbiome/assembly/'
save_directory = '/nfshomes/stakeshi/esm/data'
# Iterate through files in the directory
for filename in os.listdir(read_directory):
    if filename.endswith(".fa"):
        file_path = os.path.join(read_directory, filename)
        sequences = process_fa_file(file_path)
        
        # Create a pickle filename based on the original filename
        pickle_filename = os.path.splitext(filename)[0] + '.pkl'
        pickle_filepath = os.path.join(save_directory, pickle_filename)
        
        # Save the list as a pickle file
        with open(pickle_filepath, 'wb') as pickle_file:
            pickle.dump(sequences, pickle_file)
        print(f"Saved {pickle_filename}")

print("Processing and pickling complete.")
