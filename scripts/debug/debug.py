import numpy as np
import os

if __name__ == "__main__":
    # Create a 3D array with shape (2, 3, 4)
    contact_dir = './dataset/interx/contact'
    for contact_file in os.listdir(contact_dir):
        contact_data = np.load(os.path.join(contact_dir, contact_file))
        if len(contact_data.shape) >1:
            print(contact_file, contact_data.shape)