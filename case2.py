import os
import shutil

def copydirs(from_file, to_file):
    if not os.path.exists(to_file): # 如不存在目标目录则创建
        os.makedirs(to_file)
    files = os.listdir(from_file) # 获取文件夹中文件和目录列表
    for f in files:
        if os.path.isdir(from_file + '/' + f): # 判断是否是文件夹
            copydirs(from_file + '/' + f, to_file + '/' + f) # 递归调用本函数
        else:
            shutil.copy(from_file + '/' + f, to_file + '/' + f) # 拷贝文件

join = os.path.join
def main():
    root_dir = r'/homes/yysun/Data/case/case'
    save_root = r'/homes/yysun/Data/save_case'
    for classify in os.listdir(root_dir):
        sub_dir = join(root_dir, classify)
        for body_part in os.listdir(sub_dir):
            sub_dir_2 = join(sub_dir, body_part)
            for sequence_name in os.listdir(sub_dir_2):
                if sequence_name == 'ROI':
                    roi_dir = join(sub_dir_2, sequence_name)
                    for roi_name in os.listdir(roi_dir):
                        case_name = roi_name.split('_')[0]
                        case_name = case_name.lower()
                        roi_path = join(roi_dir, roi_name)
                        save_dir = join(save_root, classify, case_name, body_part)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        roi_save_path = join(save_dir, roi_name)
                        shutil.copyfile(roi_path, roi_save_path)
                elif sequence_name == 'DWI':
                    dwi_dir = join(sub_dir_2, sequence_name)
                    for file_name in os.listdir(dwi_dir):
                        case_name, sequence_name = file_name.split('-')#更改了sequencename
                        case_name = case_name.lower()
                        file_dir = join(dwi_dir, file_name)
                        for file_name in os.listdir(file_dir):
                            next_name = join(file_dir, file_name)
                            if os.path.isdir(next_name):
                                for file_name in os.listdir(next_name):
                                    if file_name == 'VERSION':
                                        continue
                                    final_dir = join(next_name, file_name)
                                    save_dir = join(save_root, classify, case_name, body_part, sequence_name)
                                    copydirs(final_dir, save_dir)
                else:
                    sub_dir_3 = join(sub_dir_2, sequence_name)
                    for case_name in os.listdir(sub_dir_3):
                        file_dir = join(sub_dir_3, case_name)
                        for file_name in os.listdir(file_dir):
                            next_name = join(file_dir, file_name)
                            if os.path.isdir(next_name):
                                for file_name in os.listdir(next_name):
                                    if file_name == 'VERSION':
                                        continue
                                    final_dir = join(next_name, file_name)
                                    case_name = case_name.lower()
                                    save_dir = join(save_root, classify, case_name, body_part, sequence_name)
                                    copydirs(final_dir, save_dir)


main()