import sys
import subprocess
import os
import pydicom
import numpy as np
import cv2
import argparse


# Define your function
def convert_dicom_to_png(dcm_file, outdir, extracted_string):   
    with open("output_log.txt", "a") as f:
        print("Dcm2PngCT.py: dcm file ", dcm_file, file=f)

    test_list = [f for f in os.listdir(str(dcm_file))]

    for f in test_list:
        file_extension = os.path.splitext(dcm_file + "/" + f)[1]


        # Check if the file has a specific extension
        if file_extension == ".dcm":
            ds = pydicom.read_file(dcm_file + "/" + f)  # Read DICOM image
            img = ds.pixel_array  # Get image array
        # print("img shape", img.shape)

            # Normalize the pixel array to 8-bit depth
            img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
            img_normalized = (img_normalized * 255).astype(np.uint8)

            # Convert normalized grayscale image array to RGB image array
            img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)

            # Resize the image to (256, 256)
            img_resized = cv2.resize(img_rgb, (256, 256))

            cv2.imwrite(outdir + extracted_string + "_" + f.replace('.dcm', '.png'), img_resized)  # Write PNG image

        else:
            continue



def process_directory(input_dir, output_folder):
    # Get a list of subdirectories in the input folder
    subdirectories = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    # Iterate over subdirectories
    for i, subdirectory in enumerate(subdirectories, start=1):

        # Finding the what patient number we're dealing with
        last_dash_index = subdirectory.rfind("/")  # Find the index of the last dash
        patient_number = subdirectory[last_dash_index + 1:]  # Extract the substring after the last dash

        ct_path_updated = os.path.join(input_dir, patient_number)


        for subdirectory in os.listdir(ct_path_updated):
            # Check if the item is a subdirectory
            if os.path.isdir(os.path.join(ct_path_updated, subdirectory)):
                # Access the subdirectory
                subfolder_path = os.path.join(ct_path_updated, subdirectory)                

                for subsubfolder in os.listdir(subfolder_path):
                    if os.path.isdir(os.path.join(subfolder_path, subsubfolder)) and("NA" in subsubfolder or "NLST" in subsubfolder  or "CHEST" in subsubfolder or "EASE" in subsubfolder or "BOTTOM" in subsubfolder or "Recon" in subsubfolder or "Chest" in subsubfolder or "Thor" in subsubfolder or "ACRIN" in subsubfolder):
                        subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                        convert_dicom_to_png(subsubfolder_path, output_folder,patient_number)


inputdir = "MergedImages"
outdir = "data/train/images/"
process_directory(inputdir, outdir)

