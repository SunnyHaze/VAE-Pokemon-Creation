from torch.utils.data import Dataset
import torch
import random
import numpy as np
import os
import cv2
from PIL import Image
# from . import img_reader

# class custom_dataset(Dataset):
#     def __init__(self, path, seq_len = 80, color_dim = 167):
#         # re-manage datasets, transform into one-hot tensors
#         pixel_color_data = torch.tensor(img_reader.txt_matrix_reader(path), dtype=torch.long)
        
#         id_matrix = torch.eye(color_dim)
#         self.seq_len = seq_len
#         self.one_hot_data = id_matrix[pixel_color_data] # shape: 792 400 167
        
#     def __getitem__(self, idx):
#         sample = self.one_hot_data[idx]
#         start_p_idx = torch.randint(0, 400 - self.seq_len, (1,)) # minus 1 to give space to label
#         stop_p_idx = start_p_idx + self.seq_len
#         # print(start_p_idx, stop_p_idx)
#         label = sample[stop_p_idx]
#         # print(start_p_idx, stop_p_idx, stop_p_idx)
        
#         if self.seq_len == -1 or self.seq_len < 0:
#             return sample
#         else:
#             return idx, sample[start_p_idx : stop_p_idx], label
        
#     def __len__(self):
#         return len(self.one_hot_data)
    
class figure_dataset(Dataset):
    def __init__(self, path):
        self.img_name_list = os.listdir(path)
        for i in self.img_name_list:
            assert i.endswith("png"), f"There is a image not ends with png: {i}"
        self.path = path
    def __getitem__(self, index):
        img_name = os.path.join(self.path, self.img_name_list[index])
        image = Image.open(img_name)
        image = image.convert('RGBA') # remove the alpha channel
        image = np.array(image) / 255
        image = torch.Tensor(image)  # H, W, C   C=RGBA
        alpha = image[:, :, -1].unsqueeze(2)  # (H, W, 1)
        image = image[:,:,: 3]
        image = image * alpha +  (1-alpha)
        return image.permute(2, 0, 1) # C ,H, W
    def __len__(self):
        return len(self.img_name_list)
    
if __name__ == "__main__":
    # data = custom_dataset("pixel_color.txt")
    data = figure_dataset("/mnt/data0/xiaochen/workspace/VAE-Pokemon-Creation/figure")
    
    img = data[0]
    print(img.shape)
    
    import matplotlib.pyplot as plt
    np.set_printoptions(threshold=np.inf)
    print(img.numpy().shape)
    plt.imshow(img.permute(1, 2, 0).numpy())
    
    plt.savefig("test.jpg")
    # import pdb
    # pdb.set_trace()
    