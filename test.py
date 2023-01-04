import SimpleITK as sitk
import matplotlib.pyplot as plt
from MeDIT.SaveAndLoad import LoadImage
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01

data_path = r'C:\complaire\codes\BasicTool\Document\DemoData\t2.nii.gz'
roi_path = r'C:\complaire\codes\BasicTool\Document\DemoData\nc_roi.nii.gz'
data = LoadImage(data_path, is_show_info=True)[1]
roi = LoadImage(roi_path, is_show_info=True)[1]
data = Normalize01(data)
Imshow3DArray(data,roi=roi)