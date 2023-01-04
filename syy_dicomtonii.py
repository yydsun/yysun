import dicom2nifti
import os
import SimpleITK as sitk
import numpy as np


dir_path = r'C:/complaire/data/sunyiyang/sunyiyang/test'


files_list = os.listdir(dir_path)
for file_path in files_list:
    dicom_path = dir_path + '/' + file_path
    os.mkdir(dir_path + '/' + file_path + '_nii')
    dicom2nifti.convert_directory(dicom_path, dir_path + '/' + file_path+ '_nii')