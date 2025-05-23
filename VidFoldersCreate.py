import os
import numpy as np

# Path for exported data, numpy arrays
DATA_PATH_vid = os.path.join('DATA_VIDEO')

# Actions that we try to detect
actions = np.array(['RIGHT-CornerSlice', 'LEFT-CornerSlice', 'RIGHT-SideSlice', 'LEFT-SideSlice'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 0


def runCreateFolder():
    # Create the top-level action folders if they don't exist
    for action in actions:
        os.makedirs(os.path.join(DATA_PATH_vid, action), exist_ok=True)

    # Create subfolders for each sequence in each action
    for action in actions:
        for sequence in range(start_folder, start_folder + no_sequences):
            folder_path = os.path.join(DATA_PATH_vid, action, str(sequence))
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)


if __name__ == "__main__":
    runCreateFolder()
