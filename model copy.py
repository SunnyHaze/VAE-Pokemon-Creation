import torch
# from matplotlib import pyplot as plt
# from torch.nn import functional as F
from torch import nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = in_channel,
            out_channels = out_channel,
            kernel_size = kernel_size,
            stride = stride,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            in_channels = out_channel,
            out_channels = out_channel,
            kernel_size = kernel_size,
            stride = stride,
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x_ = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += x_
        x = self.relu(x)
        

class DeResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(DeResnetBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels = in_channel,
            out_channels = out_channel,
            kernel_size = kernel_size,
            stride = stride,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.ConvTranspose2d(
            in_channels = out_channel,
            out_channels = out_channel,
            kernel_size = kernel_size,
            stride = stride,
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x_ = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += x_
        x = self.relu(x)


class VAE(nn.Module):
    def __init__(self, 
                 input_size = 20,
                 cnn_channel = 256,
                #  hidden_size= 256,
                 lattent_size = 32
                 ):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.cnn_channel = cnn_channel,
        self.lattent_size = lattent_size
        self.encoder = nn.Sequential(
            ResnetBlock(3, cnn_channel // 4, 3, 1, 1),
            ResnetBlock(cnn_channel // 4, cnn_channel // 2, 3, 1, 1), # 下采样一次
            ResnetBlock(cnn_channel // 2, cnn_channel, 3, 1, 1),     # 下采样两次
            nn.Flatten(),
            nn.Linear(
                in_features=input_size * input_size * 3 , # 300
                out_features=lattent_size * 2  # both mean & std
            ),
        )
        self.decoder_linear = nn.Linear(in_features=lattent_size, out_features=input_size * input_size * 3,)
        self.decoder_deconv = nn.Sequential(
            DeResnetBlock(cnn_channel, cnn_channel // 2, 3, 1, 1),
            DeResnetBlock(cnn_channel // 2, cnn_channel // 4, 3, 1, 1),
            ResnetBlock(cnn_channel // 2,  cnn_channel // 4, 3, 1, 1 )
        )
    
    
    
    def forward(self, img):
        B, C, H, W = img.shape
        
        lattent_vectors = self.encoder(img)
        mu, log_var = torch.chunk(lattent_vectors, chunks=2, dim=1) # split it
        
        # log_var  == $\log \sigma^2$
        # reparameterize
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        z = self.decoder_linear(z)
        decode_img = torch.reshape(z, (B, self.cnn_channel, self.input_size, self.input_size))
        decode_img = self.decoder_deconv(decode_img)
        return decode_img
    
   


if __name__ == "__main__":
    img = torch.randn((2, 3, 40, 40))
    print(img.shape)
    model = VAE(40)
   
   
   
    output = model(img)
    print(output) 