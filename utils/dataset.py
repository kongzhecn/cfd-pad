import numpy as np
import skimage.io as io
from torchvision import transforms
from torch.utils.data import Dataset


preprocess = transforms.Compose(
                            [
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ]
                            )

class LivDetDataset(Dataset):
    def __init__(self,txtpath,search,transform=preprocess):
        with open(txtpath, mode='r') as ftxt:
            pathl = ftxt.readlines()
        imgs = []
        for row in pathl:
            row = row.replace('\n','')
            cond = True
            for s in search:
                if s not in row:
                    cond = False
            if cond:
                if 'Live' in row:
                    imgs.append([row, 0])
                elif 'wood' in row.lower():
                    imgs.append([row, 1])
                elif 'ecoflex' in row.lower():
                    imgs.append([row, 2])
                elif 'body' in row.lower():
                    imgs.append([row, 3])
                else:
                    imgs.append([row, 4])
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fp, label = self.imgs[index]
        img = io.imread(fp, as_gray=True).astype(np.float32)
        # img = io.imread(fp, as_gray=True)
        if self.transform is not None:
            if len(img.shape)==2:
                img = img.reshape(img.shape[0],img.shape[1],1)
            if img.shape[-1]==1:
                img = np.tile(img,(1,1,3))
            if img.shape[-1]==4:
                img = img[:,:,:3]
            img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.imgs)