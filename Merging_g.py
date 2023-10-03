
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2
import argparse

def save_slices_as_png(dcm_path, output_directory, ct_path_updated):

    flag = True
    # Load DICOM file
    ds = pydicom.dcmread(dcm_path)

    patient_number = dcm_path.split("/")[-4]

    # Extract pixel array
    pixel_array = ds.pixel_array.astype(np.int16)

    # Get number of slices
    num_slices = pixel_array.shape[0]

    if num_slices > 100:
        flag = False
    else:
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Iterate over slices
        for i in range(num_slices):
            # Get the current slice
            current_slice = pixel_array[i]

            # Convert to binary image
            binary_slice = (current_slice > 0).astype(np.uint8) * 255

            # Resize the binary slice to (512, 512)
            resized_slice = cv2.resize(binary_slice, (256, 256))

            # Get corresponding SOP Instance UID
            sop_instance_uid = extract_sop_instance_uids_SEG(dcm_path, i)
            #print("extracted SOP UID of the raw image slice: ", sop_instance_uid)
            file_name = compare_CT_SEG(sop_instance_uid, ct_path_updated)
#            print(f"Axial Slice {i}: SOP Instance UID: {sop_instance_uid}")
            # Save the resized binary slice as PNG
            output_path = os.path.join(output_directory, f"{patient_number}_{file_name}.png")
            cv2.imwrite(output_path, resized_slice)




def extract_sop_instance_uids_SEG(dicom_file, slice_number):
    dataset = pydicom.dcmread(dicom_file, force=True)

    if dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4':
        # DICOM Segmentation Object
        slice = 0
        referenced_series_sequence = dataset.ReferencedSeriesSequence
        if referenced_series_sequence is not None:
            for series_sequence in referenced_series_sequence:
                referenced_instances_sequence = series_sequence.ReferencedInstanceSequence
                if referenced_instances_sequence is not None:
                    for instance_sequence in referenced_instances_sequence:
                        if slice == slice_number:
                            sop_instance_uid = instance_sequence.ReferencedSOPInstanceUID
                            
                            return sop_instance_uid
                        
                        slice += 1
                        
    else:
        print("Not a DICOM Segmentation Object.")
        return None

'''matches the SOP UID of every slice with a .dcm ct file'''
def compare_CT_SEG(sop_instance_uid, ct_path_updated):

    for subdirectory in os.listdir(ct_path_updated):
        # Check if the item is a subdirectory
        if os.path.isdir(os.path.join(ct_path_updated, subdirectory)):
            # Access the subdirectory
            subfolder_path = os.path.join(ct_path_updated, subdirectory)
            
            for subsubfolder in os.listdir(subfolder_path):
                if os.path.isdir(os.path.join(subfolder_path, subsubfolder)) and ("NA" in subsubfolder or "NLST" in subsubfolder or  "CHEST" in subsubfolder or "EASE" in subsubfolder or "BOTTOM" in subsubfolder or "ACRIN" in subsubfolder or "Recon" in subsubfolder or "Chest" in subsubfolder or "Thor" in subsubfolder):
                    subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                  # print("subsubfolder_path ", subsubfolder_path)
    
                    # Iterate over the DICOM files in the input folder
                    for filename in os.listdir(subsubfolder_path):
                        if filename.endswith(".dcm"):
                            
                            file_path = os.path.join(subsubfolder_path, filename)
                          
                            ds = pydicom.dcmread(file_path)
                            
                            # Check if the Image SOP UID matches the desired UID
                            if ds.SOPInstanceUID == sop_instance_uid:
                                # print("annotation SOP UID: ", ds.SOPInstanceUID )
                                # print("SOP UID of the annotation slice matches with the image SOP UID !")
                                file_name = file_path.split("/")[-1]

                                file_name_f = file_name.split(".")[0]
                                seg_name = seg_path.split("/")[-1]
                            
                                return file_name_f
                        


'''Loops through every segmentation dicom file'''
def process_directory(input_dir):
    # Get a list of subdirectories in the input folder
    subdirectories = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    

    # Iterate over subdirectories
    for i, subdirectory in enumerate(subdirectories, start=1):
       # print("subdirectory ", subdirectory)


        # Finding the what patient number we're dealing with
        last_dash_index = subdirectory.rfind("/")  # Find the index of the last dash
        patient_number = subdirectory[last_dash_index + 1:]  # Extract the substring after the last dash
       
        ct_path_updated = os.path.join(ct_path, patient_number)
        
        with open("output_log.txt", "a") as f:
            print("Merging.py: dcm path ", ct_path_updated, file=f)

        # Get the subcategory name
        subcategory = os.path.basename(subdirectory)

        # Get a list of subsubdirectories in the current subdirectory
        subsubdirectories = [f.path for f in os.scandir(subdirectory) if f.is_dir()]
        
        # Iterate over subsubdirectories
        for j, subsubdirectory in enumerate(subsubdirectories, start=1):
            
            # Get the subsubcategory name
            subsubcategory = os.path.basename(subsubdirectory)

            # Get a list of subsubsubcategory directories in the current subsubdirectory
            subsubsubdirectories = [f.path for f in os.scandir(subsubdirectory) if f.is_dir()]

            # Iterate over subsubsubdirectories
            for subsubsubcategory in subsubsubdirectories:
               # print("subsubsubcategory ", subsubsubcategory)
                # Get a list of DICOM files in the current subsubsubcategory directory
                dcm_files = [f.path for f in os.scandir(subsubsubcategory) if f.is_file() and f.name.endswith('.dcm')]


                # get the .dcm number
                for i in range(len(dcm_files)):
                    dcm_nb = dcm_files[i].split("/")[-1]
                
               # print("dcm_files ", dcm_files[0])
                # Get the base name without the extension using os.path.basename()
                file_name = os.path.basename(str(dcm_files[0]))
                file_name_without_extension = os.path.splitext(file_name)[0]
              #  print("file_name_without_extension ", file_name_without_extension)
            
                # Iterate over DICOM files
                for dcm_file in dcm_files:
                    # Process each DICOM file and save slices as PNG
                    save_slices_as_png(dcm_file, output_directory, ct_path_updated)


        #print(f"slices saved as PNG in output{i}.")



seg_path  = "MergedImages"
ct_path = "MergedImages"
output_directory = "data/train/labels/"
process_directory(seg_path)
