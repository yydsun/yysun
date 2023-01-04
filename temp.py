
import dicom2nifti
import os
import SimpleITK as sitk
import numpy as np
from MeDIT.SaveAndLoad import LoadNiiData
from MeDIT.ImageProcess import ResizeSipmleITKImage
from MeDIT.Resample import ResizeNiiFile, resampleToReferenceFile
from pathlib import Path


dir_path = r'/homes/yysun/Data/dicom/dicom'
files_list = sorted(os.listdir(dir_path))
for case_name in files_list:
    print(case_name)
    if not case_name.endswith('_nii'):
        continue
    sub_dir = os.path.join(dir_path, case_name)
    t2w_path_list = list(Path(sub_dir).rglob('*t2*'))
    if len(t2w_path_list) != 1:
        print(fr'Please check {case_name} t2w file!')
        continue
    t2w_path = str(t2w_path_list[0])

    adc_path_list = list(Path(sub_dir).rglob('*adc*'))
    if len(adc_path_list) != 1:
        print(fr'Please check {case_name} adc file!')
        continue
    adc_path = str(adc_path_list[0])

    dwi_path_list = list(Path(sub_dir).rglob('*b-2000*'))
    if len(dwi_path_list) != 1:
        print(fr'Please check {case_name} dwi file!')
        continue
    dwi_path = str(dwi_path_list[0])
    
    store_t2w_path = os.path.join(sub_dir, 't2w_resample.nii.gz')
    store_adc_path = os.path.join(sub_dir, 'adc_resample.nii.gz')
    store_dwi_path = os.path.join(sub_dir, 'dwi_resample.nii.gz')
    
    ResizeNiiFile(t2w_path, store_t2w_path, expected_resolution=(0.5, 0.5, 3.))
    resampleToReferenceFile(adc_path, store_t2w_path, store_path=store_adc_path)
    resampleToReferenceFile(dwi_path, store_t2w_path, store_path=store_dwi_path)
