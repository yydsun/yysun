from pydoc import cli
import os
import codecs
from shutil import copy, rmtree
import random
import shutil


def del_file(path_data):
    for i in os.listdir(path_data):
        file_data = path_data + "/" + i
        if os.path.isfile(file_data) == True:
            os.remove(file_data)
        else:
            del_file(file_data)


suffix = ['0000.nii.gz', '0003.nii.gz']
files = []
images_dir = "../Data/dataset/images"
annotations_dir = "../Data/dataset/annotations"
dir_path = "../Data/imagesTr"

train_file = codecs.open("../Data/dataset/train.txt", 'w')
test_file = codecs.open("../Data/dataset/test.txt", 'w')

del_file(os.path.join(images_dir, "train"))
del_file(os.path.join(images_dir, "test"))

del_file(os.path.join(annotations_dir, "train"))
del_file(os.path.join(annotations_dir, "test"))

files_list = os.listdir(dir_path)
for file_path in files_list:
    file_name_array = file_path.split("_")
    file_name = file_name_array[0]+"_"+file_name_array[1]

    if file_name not in files:
        files.append(file_name)

random.shuffle(files)

'''
for i in range(len(files)):
        shutil.copyfile(os.path.join(dir_path, files[i]+"_"+suffix[0]), os.path.join(
            images_dir, files[i]+"_"+suffix[0]))
        shutil.copyfile(os.path.join(dir_path, files[i]+"_"+suffix[1]), os.path.join(
            annotations_dir, files[i]+"_"+suffix[1]))

'''
traincon = 0
testcon = 0
for i in range(len(files)):
    if i % 10 >= 0 and i % 10 <= 6:
        shutil.copyfile(os.path.join(dir_path, files[i]+"_"+suffix[0]), os.path.join(
            os.path.join(images_dir, "train"), files[i]+"_"+suffix[0]))
        shutil.copyfile(os.path.join(dir_path, files[i]+"_"+suffix[1]), os.path.join(
            os.path.join(annotations_dir, "train"), files[i]+"_"+suffix[1]))
        train_file.write("{0} {1}\n".format(os.path.join(os.path.join(images_dir, "train"),
                         files[i]+"_"+suffix[0]), os.path.join(os.path.join(annotations_dir, "train"), files[i]+"_"+suffix[1])))
        traincon += 1
    else:
        shutil.copyfile(os.path.join(dir_path, files[i]+"_" + suffix[0]), os.path.join(
            os.path.join(images_dir, "test"), files[i]+"_"+suffix[0]))
        shutil.copyfile(os.path.join(dir_path, files[i]+"_" + suffix[1]), os.path.join(
            os.path.join(annotations_dir, "test"), files[i]+"_"+suffix[1]))
        test_file.write("{0} {1}\n".format(os.path.join(os.path.join(images_dir, "test"),
                         files[i]+"_"+suffix[0]), os.path.join(os.path.join(annotations_dir, "test"), files[i]+"_"+suffix[1])))
        testcon += 1
    if i % 100 == 0:
        print("progressing {}%".format(i/len(files)*100))
train_file.close()
test_file.close()
print(traincon, testcon)
