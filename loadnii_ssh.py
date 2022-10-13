from pydoc import cli
import paramiko
import os

#读取远程服务器中的图像文件,并保存名称

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('219.228.149.72', 22, 'yysun', 'Syydasheng720', compress=True)

fullpathArry = []
skipArry = ['0000.nii.gz','0003.nii.gz']

s = paramiko.Transport(('219.228.149.72', 22))
s.connect(username = 'yysun', password = 'Syydasheng720')
sftp = paramiko.SFTPClient.from_transport(s)

sftp_client = client.open_sftp()
sftp.chdir("/homes/yqyi/nnUNet_raw_data_base/nnUNet_raw_data/Task807_Prostate_csPCa_crop/imagesTr")
pathlist = sftp.listdir(path = '.')
for i in pathlist:
    j = i.split("_")
    if j[2] in skipArry:
        fullpath = "/homes/yqyi/nnUNet_raw_data_base/nnUNet_raw_data/Task807_Prostate_csPCa_crop/imagesTr" + '/' + i 
        fullpathArry.append(fullpath)

print(fullpathArry)
print(len(fullpathArry))


sftp_client.close()
client.close()