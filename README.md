# TumorSegMedSam
This repository includes code and a description of a novel tumor segmentation algorithm by fine-tuning Meta's foundational model MedSAM on the publicly available dataset LIDC-IDRI. The code for this project is available upon request.

## Preprocessing 
In order to make the publicly available LIDC-IDRI dataset compatible with Meta's foundational model MedSAM, preprocessing was required. Multiple issues had to be tackled:
- The transformation of the 3D lung dicom file to 2D images
- Filter any 'closed' lung images. In other words, removing the lung image slices at the beginning and the end of the slice where the lung begins to close. 
- Matching the lung scan images to their corresponding annotation (labels).
- Balancing the dataset to contain 60 % of the paired lung slices and annotations to not contain tumours and 40 % to contain tumours.

## Visualization
Below is a comparaison of the performance the MedSAM before and after being fine-tuned on a subset of the LIDC-IDRI data (240 lung slices, about 2.5% of the total dataset).
The image of the left is the ground truth. The image in the middle is a lung slice passed in MedSAM with no finetuning and results in a dice coefficent of 0.6. The image on the right is the same lung slice passed in the fine-tuned MedSAM. The dice coefficient significantly improved and has now reached 0.89. 

## Preliminary results
I started by training in on a subset of the LIDC-IDRI dataset for tumours larger than 14mm which resulted in a dataset of 550 lung slices. After performing 5 fold cross validation, I found that the model performs with an average of 0.893 dice coefficient.

## Next Steps
I am currently working on trainin MedSAM on the full lidcidri dataset with tumors larger or equal to 3mm which represents about 10 500 lung images.

## Awknowledgments
- Thank you to Meta AI for making the foundational model MedSAM publically available. The link to its official [repository]([url](https://github.com/bowang-lab/MedSAM)https://github.com/bowang-lab/MedSAM)
- I am also grateful to have been able to use the open-source Lung Image Database Consortium image collection (LIDC-IDRI) to finetune this model. Access the dataset [here]([url](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)

