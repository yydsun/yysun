import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import numpy as np
import sys
sys.path.append('/homes/yysun/Codes/BasicTool')
from MeDIT.SaveAndLoad import LoadImage

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        img_name = self.images_list[index]
        case_name = img_name[:-len('_0000.nii.gz')]
        img_path = os.path.join(self.image_dir, img_name)
        image = LoadImage(img_path)[1]
        mask_name = case_name+ '_0003.nii.gz'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = LoadImage(mask_path)[1]

        if image.shape[2] < 20:
            pad_image = np.zeros((image.shape[0], image.shape[1], 20-image.shape[2]))
            image = np.concatenate((image, pad_image), axis=-1)
        if self.transform is not None:
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return np.array(image, dtype=np.float32), np.array(mask, dtype=np.float32)
                   

        
if __name__ == '__main__':
    image_dir = r'/homes/yysun/Data/dataset/images/train'
    mask_dir = r'/homes/yysun/Data/dataset/annotations/train'
    dataset_train = CarvanaDataset(image_dir, mask_dir)
    train_loader = DataLoader(
        dataset_train,
        batch_size = 8,
        shuffle = True,
    )
    import matplotlib.pyplot as plt
    id = 0
    for image, mask in train_loader:
        image = np.array(image)
        mask = np.array(mask)
        for i in range(image.shape[0]):
            i_image = image[i, :, :, :]
            i_mask = mask[i, :, :, :]
            slice = np.argmax(np.sum(i_image, axis=(0,1)))
            image_silce = i_image[:, :, slice]
            mask_slice = i_mask[:, :, slice]
            plt.imshow(image_silce, cmap='gray')
            plt.contour(mask_slice, colors='r')
            plt.savefig(fr'/homes/yysun/Data/dataset/jpg/{id}.jpg', dpi=400)
            plt.close()
            id += 1