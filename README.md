# Description 
This repository provides code and describes a deep learning tumor segmentation model I developed by fine-tuning Meta's foundational model MedSAM on the publicly available dataset LIDC-IDRI. This project is a part of a larger diagnostic pipeline designed to be used by UHN Princess Margaret Cancer Centre.

## Preprocessing 
In order to make the publicly available LIDC-IDRI dataset compatible with Meta's foundational model MedSAM, preprocessing was required. Multiple issues had to be tackled:
- The transformation of the 3D lung dicom file to 2D images.
- The transformation of the 2D lung tumor annotations to 2D images.
- Matching the lung scan images to their corresponding annotations using the dicom's metadata.

## Visualization
Below is a comparaison of the performance of the MedSAM model before and after being fine-tuned on a subset of the LIDC-IDRI data. The subset of the data included 240 lung slices, about 2.5% of the total dataset.
![visualization](https://github.com/charlottevedrines/TumorSegMedSam/assets/97196465/83b0eb68-cd5d-47ce-b387-be0446a88778)
- The image of the left is the ground truth.
- The image in the middle is a lung slice passed in MedSAM model that did not undergo finetuning. The resulting dice coefficent of 0.287.
- The image on the right is the same lung slice passed into the fine-tuned MedSAM model. The dice coefficient significantly improved and has now reached 0.873. 

## Preliminary results
I started by training MedSam on a subset of the LIDC-IDRI dataset. This subset of data only included tumours larger than 14mm which resulted in a dataset of 550 lung slices. After performing 5 fold cross validation, I found that the model performs with an average of 0.893 dice coefficient. These results seem quite promising for my next step which is to train MedSAM on the full lidcidri dataset with tumors larger or equal to 3mm which represents about 10 500 lung images. 

These results where obtained by only training the model on slices of the 3D lung scan that contained tumours. However, I ultimately want the model to take in as input the 3D lung scan which will also include lung slices that do not contain any tumors.

## Next Steps
### Further preprocessing
- As mentioned above, the goal is that the model performs well on lung slices that contain and do not contain tumors. I am working on balancing the dataset to contain 60 % of the paired lung slices and annotations with no tumours and 40 % to contain tumours.
- Filter any 'closed' lung images which reside at the beginning and the end of the slice where the lung begins to close.
  
### Widening the scope
I am currently working on training MedSAM on the full lidcidri dataset with tumors larger or equal to 3mm which represents about 10 500 lung images.

## Awknowledgments
- Thank you to Meta AI for making the foundational model MedSAM publically available. The link to its official [repository]([url](https://github.com/bowang-lab/MedSAM)https://github.com/bowang-lab/MedSAM)
- I am also grateful to have been able to use the open-source Lung Image Database Consortium image collection (LIDC-IDRI) to finetune this model. Access the dataset [here]([url](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)

# Running the code
This code will run on cpu. Change ```pre_gre_rgb2D.py``` and ```DL_model.py``` appropriately to run this model on GPU.

## Installation
1. Create a virtual environment ```conda create -n medsamtumour python=3.10 -y``` and activate it ```conda activate medsamtumour```
2. ```Install Pytorch 2.0```
3. ```pip install monai```
4. ```git clone https://github.com/charlottevedrines/TumorSegMedSam```
5. Enter the MedSAM folder ```cd MedSAM``` and run ```pip install -e```

## Running the model
Download the[model checkpoint]([url](https://drive.google.com/file/d/1tKd7p3cLVzvF3B4fpopijwNo2LSbKNWV/view?usp=drive_link)) and place it in ```work_dir/SAM/```

To start, run the script ```CentralScript_g.py```. This will run the model on a sample of the LIDC-IDRI dataset included in this repository.

