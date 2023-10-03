import os
import shutil
import sys
import subprocess


folder1 = "data/train/labels"
folder2 = "data/train/images"

# Get the list of image files in folder1
files_folder1 = [f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]

# Iterate over the image files in folder2
for file2 in os.listdir(folder2):
   
    if file2.lower().endswith(".png"):
       file2_path = os.path.join(folder2, file2)

       # Extract the title by removing the file extension
       file2_title = file2_path.rsplit('/', 1)[-1]

       # Check if the corresponding image file exists in folder1

       if file2_title in files_folder1:
            # Remove the file if it doesn't have a corresponding image in folder1
           continue
       else:
           os.remove(file2_path)
        
  
