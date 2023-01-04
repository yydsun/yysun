# -*- coding: utf-8 -*-
import dicom2nifti
import os
import SimpleITK as sitk
import numpy as np

def resample_image(image, expected_resolution=[0.5, 0.5, 3.],  expected_shape=None):
    resolution = image.GetSpacing()
    shape = image.GetSize()
 
    # 根据输出expected_resolution设置新的size
    if expected_resolution is None:
        dim_0, dim_1, dim_2 = False, False, False
        if expected_shape[0] < 1e-6:
            expected_shape[0] = shape[0]
            dim_0 = True
        if expected_shape[1] < 1e-6:
            expected_shape[1] = shape[1]
            dim_1 = True
        if expected_shape[2] < 1e-6:
            expected_shape[2] = shape[2]
            dim_2 = True
        expected_resolution = [raw_resolution * raw_size / dest_size for dest_size, raw_size, raw_resolution in
                               zip(expected_shape, shape, resolution)]
        if dim_0: expected_resolution[0] = resolution[0]
        if dim_1: expected_resolution[1] = resolution[1]
        if dim_2: expected_resolution[2] = resolution[2]
    
    elif expected_shape is None:
        dim_0, dim_1, dim_2 = False, False, False
        if expected_resolution[0] < 1e-6: 
            expected_resolution[0] = resolution[0]
            dim_0 = True
        if expected_resolution[1] < 1e-6: 
            expected_resolution[1] = resolution[1]
            dim_1 = True
        if expected_resolution[2] < 1e-6: 
            expected_resolution[2] = resolution[2]
            dim_2 = True
        expected_shape = [int(raw_resolution * raw_size / dest_resolution) for
                       dest_resolution, raw_size, raw_resolution in zip(expected_resolution, shape, resolution)]
        if dim_0: expected_shape[0] = shape[0]
        if dim_1: expected_shape[1] = shape[1]
        if dim_2: expected_shape[2] = shape[2]
   
 
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(expected_resolution)
    resample.SetSize(expected_shape)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
 
    resample.SetInterpolator(sitk.sitkBSpline)
 
    return  sitk.ImageFileReader.Execute(image)

from MeDIT.SaveAndLoad import LoadNiiData
from MeDIT.ImageProcess import ResizeNiiFile

dir_path = r'Data/dicom/dicom'
'''
if __name__ == '__main__':
    files_list = os.listdir(dir_path)
    for file_path in files_list:
        dicom_path = r"C:/Complaire/Data/yysun/dicom" + '/' + file_path
        os.mkdir(r'C:/Complaire/Data/yysun/dicom' + '/' + file_path + '_nii')
        dicom2nifti.convert_directory(dicom_path,r'C:/Complaire/Data/yysun/dicom' + '/' + file_path+ '_nii')
'''
files = []
files_list = os.listdir(dir_path)
for file_path in files_list:
    file_name_array = file_path.split("_")
    if len(file_name_array) > 1:
        file_name = file_name_array[0]+"_"+file_name_array[1]
        files.append(file_name)
        #files为数字_nii文件夹名字集合
#遍历所有目标文件夹，得到文件名与文件路径
for file in files:
    #先处理一个file里的nii图像
    file_list = os.listdir(r'Data/dicom/dicom' + '/' + file)
    for nii_name in file_list: 
        
        image_path = r'Data/dicom/dicom' + '/' + file + '/' + nii_name
        #store_path = r'Data/dicom/dicom' + '/' + file  + '/' + '_re_'  + nii_name
        #os.mkdir(store_path)
        
        Original_img = sitk.ReadImage(image_path)
        Resample_img = resample_image(Original_img)
        #sitk.WriteImage(Resample_img, store_path)
        