import torch
from model import VAE
from utils import data_utils
import matplotlib.pyplot as plt
import numpy as np
ckpt_path = "output_dir_lr5e-4_epoch10000_color_latent16/checkpoint_9600.pt"

dataset = data_utils.figure_dataset("./figure")


state_dict = torch.load(ckpt_path)

# Trace settings from the ckpt
linear_shape = state_dict['encoder.1.weight'].shape  # (1024, 4800) = (1024, 40 * 40 * 3)
input_size = int((linear_shape[1] / 3) ** 0.5)
hidden_size = int(linear_shape[0])
latent_size = state_dict['encoder.9.weight'].shape[0] // 2


# print(input_size)
print(input_size)
model = VAE(input_size=input_size, hidden_size=hidden_size, lattent_size=latent_size)

model.load_state_dict(state_dict=state_dict, strict=True)


max_img_num = 4

step = 21
grid_bound = 1.5
selectd_dim = [2, 6]

assert selectd_dim[0] < latent_size and selectd_dim[1] < latent_size, "selectd_dimesion should smaller than the latent_size"
    


img1 = dataset[0]
plt.subplot(1, max_img_num, 1)
plt.title("raw input")
plt.imshow(img1.permute(1, 2, 0).numpy())

img1 = img1.unsqueeze(0) # batching

with torch.no_grad():
    z, mu, _  = model.encode(img1)
    print(z.shape)
    print(z)
    print(mu)


    recon_img = model.decode(z)
    plt.subplot(1, max_img_num, 2)
    plt.title("reconstruct Z")
    plt.imshow(recon_img[0].permute(1, 2, 0).numpy())


    print(mu.shape)
    # mu[0][0] += 1
    # mu[0][1] += 3
    # mu[0][3] += 3
    mu[0][0] += 0.4
    mu[0][1] += 0.5
    
    
    recon_img = model.decode(mu)
    plt.subplot(1, max_img_num, 3)
    plt.title("reconstruct $mu$")
    plt.imshow(recon_img[0].permute(1, 2, 0).numpy())

    rand_v= torch.randn_like(z)
    recon_img = model.decode(rand_v)
    plt.subplot(1, max_img_num, 4)
    plt.title("random z")
    plt.imshow(recon_img[0].permute(1, 2, 0).numpy())
    
    
    
    # first save
    plt.savefig("test_visual.png")


    # grid visualize
    
    plt.cla()
    plt.clf()
    


    
    # Torch has a deprecated issue, check here: https://pytorch.org/docs/stable/generated/torch.meshgrid.html#torch-meshgrid
    latent_grid = torch.tensor(np.array(np.meshgrid(       # create a gird for VAE grid visualize
        np.linspace(-grid_bound, grid_bound , num=step),
        np.linspace(-grid_bound, grid_bound , num=step),
    )))
    
    # print(latent_grid)
    # latent_grid_shape ==  2, 10, 10
    latent_grid = latent_grid.reshape(2, step * step)
    latent_grid = latent_grid.permute(1, 0)
    print(latent_grid.shape)
    
    # z_tensor = torch.repeat_interleave(z, 100, 0) # 100, 16
    z_tensor = torch.repeat_interleave(mu, step * step, 0) # 100, 16
    
    z_tensor[:, selectd_dim[0]] += latent_grid[:, 0]
    z_tensor[:, selectd_dim[1]] += latent_grid[:, 1]
    
    print(z_tensor.shape)
    z_img = model.decode(z_tensor)  # no batch, may OOM
    print(z_img.shape)
    
    plt.figure(figsize=(20, 20), dpi=300)  # Increase figure size and set dpi for higher resolution
    
    for i in range(step * step):
        plt.subplot(step, step, i + 1)
        plt.imshow(z_img[i].permute(1,2,0).numpy())
        plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)  # Set spacing between subplots to zero
    
    file_name = ckpt_path.replace('/',"-")
    plt.savefig(f"grid_img_{file_name}_step{step}_bound{grid_bound}_dim{selectd_dim}.png")
        

    



# print(model)

