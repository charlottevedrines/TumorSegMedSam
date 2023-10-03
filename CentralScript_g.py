import subprocess 
import random
import shutil
import os
import sys
import platform
import monai

print("Starting the CentralScript.\n")


def k_fold_cross_validation(images_folder, labels_folder, k):
    # Get lists of all image files in the images and labels folders
    image_files = [f for f in os.listdir(images_folder) if f.endswith(".png")]
    label_files = [f for f in os.listdir(labels_folder) if f.endswith(".png")]
    
    # Sort the lists to ensure consistent shuffling and matching of images and labels
    image_files.sort()
    label_files.sort()
    
    # Randomly shuffle the image and label files
    random.shuffle(image_files)
    random.shuffle(label_files)
    
    fold_size = len(image_files) // k  # Calculate the size of each fold
    
    folds = []  # List to store the generated folds
    
    for i in range(k):
        start = i * fold_size  # Starting index of the current fold
        end = start + fold_size  # Ending index of the current fold
        
        if i == k - 1:  # Adjust the end index for the last fold if necessary
 
           end = len(image_files)
        
        val_image_files = image_files[start:end]  # Image files for the validation set
        val_label_files = label_files[start:end]  # Label files for the validation set
        
        train_image_files = image_files[:start] + image_files[end:]  # Image files for the training set
        train_label_files = label_files[:start] + label_files[end:]  # Label files for the training set
        
        # Create the train and validation folders for the current fold
        train_folder_images =  f"k_fold_cv/train/images/fold_{i + 1}"
        train_folder_labels =  f"k_fold_cv/train/labels/fold_{i + 1}"
        val_folder_images = f"k_fold_cv/test/images/fold_{i + 1}"
        val_folder_labels =  f"k_fold_cv/test/labels/fold_{i + 1}"
        
        # Create the train and validation folders if they don't exist
        os.makedirs(train_folder_images, exist_ok=True)
        os.makedirs(train_folder_labels, exist_ok=True)
        os.makedirs(val_folder_images, exist_ok=True)
        os.makedirs(val_folder_labels, exist_ok=True)
        
        # Copy raw images and their corresponding labeled images to the respective folders
        for image_file in train_image_files:
            src_image_path = os.path.join(images_folder, image_file)
            src_label_file = image_file
            src_label_path = os.path.join(labels_folder, src_label_file)
            if os.path.exists(src_label_path):
                dst_image_path = os.path.join(train_folder_images, image_file)
                dst_label_path = os.path.join(train_folder_labels, src_label_file)
                shutil.copy(src_image_path, dst_image_path)
                shutil.copy(src_label_path, dst_label_path)
        
        for image_file in val_image_files:
            src_image_path = os.path.join(images_folder, image_file)
            src_label_file = image_file
            src_label_path = os.path.join(labels_folder, src_label_file)
            if os.path.exists(src_label_path):
                dst_image_path = os.path.join(val_folder_images, image_file)
                dst_label_path = os.path.join(val_folder_labels, src_label_file)
                shutil.copy(src_image_path, dst_image_path)
                shutil.copy(src_label_path, dst_label_path)
        
        folds.append((train_folder_images, train_folder_labels, val_folder_images, val_folder_labels))  # Append the fold to the list
    
    return folds

# Usage example
images_folder = "data/train/images"  # Folder containing the raw image files (.png)
labels_folder = "data/train/labels"  # Folder containing the labeled image files (.png)
k = 5  # Number of folds

commandDcm2Png = "python3 Dcm2PngCT_g.py"

# Capture the subprocess output
#completed_process = subprocess.run(commandDcm2Png, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

#Write the captured output to the output_log.txt file
# with open("output_log.txt", "a") as output_file:
#    output_file.write(completed_process.stdout)
#    output_file.write(completed_process.stderr)


commandMerging = "python3 Merging_g.py"
#subprocess.run(commandMerging, shell=True)

commandRemovingPng = "python3 RemovingPng_g.py"
#subprocess.run(commandRemovingPng, shell=True)

folds = k_fold_cross_validation(images_folder, labels_folder, k)

#Print the training and validation folders for each fold
for i, (train_images, train_labels, val_images, val_labels) in enumerate(folds):
    print(f"Fold {i + 1}:")
    print("Training images folder:", train_images)
    print("Training labels folder:", train_labels)
    print("Validation images folder:", val_images)
    print("Validation labels folder:", val_labels)

    print("Inside CentralScript.py Running on machine:", platform.node())

    #subprocess.run(["python3", "pre_grey_rgb2D.py", "-i", train_images, "-gt", train_labels])
    print("Starting the DL_model.\n")
    subprocess.run(["python3", "DL_model.py", val_images, val_labels])




