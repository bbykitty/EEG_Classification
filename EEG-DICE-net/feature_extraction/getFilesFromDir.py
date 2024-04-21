import os

def find_files(directory_path, filetype):
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(filetype):
                all_files.append(os.path.join(root, file))
    return all_files

# Example usage:
directory_path = "/s/chopin/k/grad/mbrad/cs535/EEG_Classification/derivatives"
set_files = find_files(directory_path, ".set")
for set_file in set_files:
    print(set_file)
