
# This is going to be a script that you guys can copy and paste to test our numbers on
# the machine learning models


import numpy as np
from PIL import Image

def load_and_process_image(image_path):
    with Image.open(image_path) as img:
        img_array = np.array(img)  # Convert the image to a numpy array
        if len(img_array.shape) == 3:  # Check if the image has 3 dimensions
            img_2d = img_array[:,:,0]  # Convert to 2D if necessary
        else:
            img_2d = img_array  # Use the original array if it's already 2D
        return img_2d

# Replace with the path to your image
image_path = r'C:\Users\user\Desktop\School\MSAAI_501\Group_Projects\Sydneys_Nums_Formatted\4.jpg'

# processing the image
image_data = load_and_process_image(image_path)

print(image_data.shape)


## Youll then run the predict function on the 'image_data' variable

#%%
