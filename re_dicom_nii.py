
import dicom2nifti
import os
import SimpleITK as sitk
import numpy as np
from MeDIT.SaveAndLoad import LoadNiiData
from MeDIT.ImageProcess import ResizeSipmleITKImage
from MeDIT.Resample import ResizeNiiFile
'''
def resample_image(image, expected_resolution=[0.5, 0.5, 3.],  expected_shape=None):
    resolution = image.GetSpacing()
    shape = image.GetSize()
 
    # 根据输出expected_resolution设置新的size
    expected_shape = [
        int(np.round(shape[0] * resolution[0] / expected_resolution[0])),
        int(np.round(shape[1] * resolution[1] / expected_resolution[1])),
        int(np.round(shape[2] * resolution[2] / expected_resolution[2]))
    ]
   
 
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(expected_resolution)
    resample.SetSize(expected_shape)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
 
    resample.SetInterpolator(sitk.sitkBSpline)
 
    return  resample.Execute(image)
'''

dir_path = r'/homes/yysun/Data/dicom/dicom'
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
    file_list = os.listdir(dir_path + '/' + file)
    for nii_name in file_list: 
        print(nii_name)
        image_path = dir_path + '/' + file + '/' + nii_name
        store_path = dir_path + '/' + file  + '/' + '_re_'  + nii_name
        #os.mkdir(store_path)
        
        image, _, _ = LoadNiiData(image_path)
        
        ResizeNiiFile(image_path, store_path, expected_resolution=(0.5, 0.5, 3.))

        '''
        resample_rs = resample_image(sitk.ReadImage(image_path))
        sitk.WriteImage(resample_rs, store_path)
        Original_img = sitk.ReadImage(image_path)
        Resample_img = resample_image(Original_img)
        sitk.WriteImage(Resample_img, store_path)
        '''