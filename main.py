# from datasets.img_reader import colormap, pixel_color
import os
import argparse
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch import nn as nn
from torch import optim as optim 
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from utils.data_utils import figure_dataset
from utils.img_reader import mapping_img, txt_matrix_reader, colormap
from utils.lr_sched  import adjust_learning_rate
import numpy as np


from model import VAE

def get_args_parser():
    parser = argparse.ArgumentParser('Pokemon Generator by LSTM Training', add_help=True)
    parser.add_argument("--batch_size", default=256, type=int,
                        help="Batchsize per GPU")
    parser.add_argument("--seq_len", default=100, type=int,
                        help="Batchsize per GPU")
    parser.add_argument("--output_dir", default="output_dir_lr1e-3_epoch5000", type= str,
    # parser.add_argument("--output_dir", default="out_test", type= str,
                        help = 'output dir for ckpt and logs')
    parser.add_argument("--epoch", default=5000, type=int,
                        help = 'Number of epochs')
    parser.add_argument("--lr", default="1e-3", type=float,
                        help = 'Learning rate')
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device: cuda or GPU")
    parser.add_argument("--test_period", default=200, type=int,
                        help = 'Test when go through this epochs')
    parser.add_argument("--mask_rate", default=0.5, type=int,
                        help = 'masked rate of the input images')
    parser.add_argument("--save_period", default=200, type=int,
                        help = 'masked rate of the input images')
    parser.add_argument("--warmup_epochs", default=20, type=int,
                        help = 'warmup epochs')
    parser.add_argument("--min_lr", default=1e-7, type=float,
                        help = 'min lr for lr schedule')
    parser.add_argument("--seed", default=41, type=int,
                        help = 'random seed init')

    return parser

# 计算VAE的损失函数
def vae_loss(reconstructed_x, x, mu, log_var):
    # print(torch.max(x), torch.min(x))
    # print(torch.max(reconstructed_x), torch.min(reconstructed_x))/
    reconstruction_loss = F.binary_cross_entropy(reconstructed_x, x, reduction="sum")
    # reconstruction_loss = F.binary_cross_entropy(reconstructed_x, x)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence

def main(args):
    # torch.manual_seed(114514)
    torch.manual_seed(args.seed)
    log_writer = SummaryWriter(log_dir=args.output_dir)
    
    my_dataset = figure_dataset("/mnt/data0/xiaochen/workspace/VAE-Pokemon-Creation/figure")

    train_ratio = 0.99
    test_ratio = 1 - train_ratio

    train_size = int(train_ratio * len(my_dataset))
    test_size = len(my_dataset) - train_size

    train_dataset, test_dataset = random_split(my_dataset, [train_size, test_size])

    print(len(train_dataset))
    print(len(test_dataset))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size = args.batch_size, 
        shuffle=True,
        drop_last=True
    )


    device = args.device
    model = VAE(40)
    
    model = model.to(device)
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    # hidden state for LSTM
    # hidden = model.init_hidden(args.batch_size, device = device)
    # hidden = hidden.to(device)

    # ----previsualize the test images:----
    test_rgb_images = []
    test_rgb_images_masked = []
    # for idx, _, _ in test_dataset:
    #     img = whole_dataset[idx]
    #     pixel_img = torch.argmax(img, dim=-1)
    #     rgb_img = torch.tensor(mapping_img(pixel_img))
        
    #     visible_height = int(args.mask_rate * 20)
    #     masked_img = torch.zeros_like(rgb_img)
    #     masked_img[0:visible_height, :, : ] = rgb_img[0:visible_height, :, : ]
    #     test_rgb_images.append(rgb_img)      
    #     test_rgb_images_masked.append(masked_img)
    # exit(0)
    
    # print(rgb_img.shape)
    # test_rgb_img = torch.cat(test_rgb_images, dim=1)
    # print(test_rgb_img.shape)
    # test_rgb_img_masked = torch.cat(test_rgb_images_masked, dim=1)
    
    # plt.imsave("test.png", np.uint8(test_rgb_img.numpy()))
    # log_writer.add_images("test_set/raw", test_rgb_img.permute(2, 0, 1).unsqueeze(0) / 256)
    # log_writer.add_images("test_set/masked", test_rgb_img_masked.permute(2, 0, 1).unsqueeze(0) / 256)
    # log_writer.flush()
    # exit(0)
    #  -----------------------------------
    for epoch in range(args.epoch):
        running_loss = 0.0
                
        # exit(0)
        # train for one epoch
        model.train()
        lr = adjust_learning_rate(optimizer=optimizer, epoch=epoch, args=args)
        log_writer.add_scalar("train/lr", lr, epoch)
        for data in train_loader:
            data = data.to(device)
            # print(data.shape)
            optimizer.zero_grad()
            predict_img, mu, log_var = model(data)
            # print(predict_img.shape)
            predict_img = torch.sigmoid(predict_img)
            loss = vae_loss(
                    reconstructed_x=predict_img,
                    x = data,
                    mu=mu,
                    log_var=log_var
                )
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss
        log_writer.add_scalar("train/epoch_loss", running_loss / len(train_loader), epoch)
        if epoch % 50 == 0 or epoch == args.epoch:
            log_writer.add_images("train/x", data[:16], epoch)
            log_writer.add_images("train/reconstruct_img", predict_img[:16], epoch)
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
        log_writer.flush()
        
        if epoch % args.save_period == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_{epoch}.pt"))
        
if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    
    
# # re-manage datasets, transform into one-hot tensors
# pixel_color_data = torch.tensor(pixel_color, dtype=torch.long)
# print(pixel_color_data.shape)
# one_hot_matrix = torch.eye(167)

# one_hot_imgs = one_hot_matrix[pixel_color_data]
# print(one_hot_imgs.shape)

# print(one_hot_imgs[0])


# pixel_imgs = torch.argmax(one_hot_imgs, dim=-1)
# print(pixel_imgs.shape)
# print(torch.sum(pixel_imgs - pixel_color_data)) 



# batchsize = 4


# model = PixelRNN()
# print(model)

# hidden = model.init_hidden(1)
# train_X = one_hot_imgs[0:1]
# print(train_X.shape)
# output, hidden = model(train_X, hidden)
# print(output.shape)
# print(output)