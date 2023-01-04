import os
import shutil
#处理neck
def copydirs(from_file, to_file):
    if not os.path.exists(to_file): # 如不存在目标目录则创建
        os.makedirs(to_file)
    files = os.listdir(from_file) # 获取文件夹中文件和目录列表
    for f in files:
        if os.path.isdir(from_file + '/' + f): # 判断是否是文件夹
            copydirs(from_file + '/' + f, to_file + '/' + f) # 递归调用本函数
        else:
            shutil.copy(from_file + '/' + f, to_file + '/' + f) # 拷贝文件

pic0_neck_path = r'/homes/yysun/Data/case/case/0/neck'
pic0_path = r'/homes/yysun/Data/case/case/0'

for root, dirs, files in os.walk(pic0_neck_path, topdown=False):
    for dir_name in dirs:
        if dir_name == 'CE.T1':
            for i in os.listdir(os.path.join(pic0_neck_path, dir_name)):
                print(i)                
                copydirs(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, i, 'neck', dir_name))
                    
        if dir_name == 'T2':
            for i in os.listdir(os.path.join(pic0_neck_path, dir_name)):
                print(i)
                copydirs(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, i, 'neck', dir_name))

        if dir_name == 'ROI':
            for i in os.listdir(os.path.join(pic0_neck_path, dir_name)):
                if i == 'p001_c.nii.gz':
                    file_dir = os.path.join(pic0_path, 'p001', 'neck', dir_name)
                    if not os.path.exists(file_dir): # 如不存在目标目录则创建
                        os.makedirs(file_dir)
                    #移动并改名
                    shutil.copy(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, 'p001', 'neck', dir_name)) # 拷贝文件
                elif i == 'p002_c.nii.gz':
                    file_dir2 = os.path.join(pic0_path, 'p002', 'neck', dir_name)
                    if not os.path.exists(file_dir2): # 如不存在目标目录则创建
                        os.makedirs(file_dir2)
                    shutil.copy(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, 'p002', 'neck', dir_name))
                elif i == 'p003_c.nii.gz':
                    file_dir3 = os.path.join(pic0_path, 'p003', 'neck', dir_name)
                    if not os.path.exists(file_dir3): # 如不存在目标目录则创建
                        os.makedirs(file_dir3)
                    shutil.copy(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, 'p003', 'neck', dir_name))
                   
pic0_nasopharynx_path = r'/homes/yysun/Data/case/case/0/nasopharynx'
pic0_path = r'/homes/yysun/Data/case/case/0'

for root, dirs, files in os.walk(pic0_neck_path, topdown=False):
    for dir_name in dirs:
        if dir_name == 'CE.T1':
            for i in os.listdir(os.path.join(pic0_neck_path, dir_name)):
                print(i)
                if i == 'p001':
                    #创建并移动
                    copydirs(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, i, 'neck', dir_name))
                    print('ok1')

                elif i == 'p002':
                    copydirs(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, i, 'neck', dir_name))
                    print('ok2')
                elif i == 'p003':
                    copydirs(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, i, 'neck', dir_name))
                    print('ok3')
        if dir_name == 'T2':
            for i in os.listdir(os.path.join(pic0_neck_path, dir_name)):
                print(i)
                if i == 'p001':
                    #移动并改名
                    copydirs(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, i, 'neck', dir_name))
                    print('ok1')
                elif i == 'p002':
                    copydirs(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, i, 'neck', dir_name))
                    print('ok2')
                elif i == 'p003':
                    copydirs(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, i, 'neck', dir_name))
                    print('ok3')
        if dir_name == 'ROI':
            for i in os.listdir(os.path.join(pic0_neck_path, dir_name)):
                if i == 'p001_c.nii.gz':
                    file_dir = os.path.join(pic0_path, 'p001', 'neck', dir_name)
                    if not os.path.exists(file_dir): # 如不存在目标目录则创建
                        os.makedirs(file_dir)
                    #移动并改名
                    shutil.copy(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, 'p001', 'neck', dir_name)) # 拷贝文件
                elif i == 'p002_c.nii.gz':
                    file_dir2 = os.path.join(pic0_path, 'p002', 'neck', dir_name)
                    if not os.path.exists(file_dir2): # 如不存在目标目录则创建
                        os.makedirs(file_dir2)
                    shutil.copy(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, 'p002', 'neck', dir_name))
                elif i == 'p003_c.nii.gz':
                    file_dir3 = os.path.join(pic0_path, 'p003', 'neck', dir_name)
                    if not os.path.exists(file_dir3): # 如不存在目标目录则创建
                        os.makedirs(file_dir3)
                    shutil.copy(os.path.join(pic0_neck_path, dir_name, i), os.path.join(pic0_path, 'p003', 'neck', dir_name))
                    #需要重命名，问一下为什么不行