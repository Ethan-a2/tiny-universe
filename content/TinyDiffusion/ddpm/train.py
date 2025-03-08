import os
import numpy as np  # 添加这一行
import matplotlib.pyplot as plt;

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm

from diffusion import NoiseScheduler
from unet import SimpleUnet
from dataloader import load_transformed_dataset
from sample import sample, plot


def test_step(model, dataloader, noise_scheduler, criterion, epoch, num_epochs, device):
    """测试步骤,计算测试集上的损失"""
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        num_batches = 0
        pbar = tqdm(dataloader)
        for batch in pbar:
            images, _ = batch
            images = images.to(device)
            t = torch.full((images.shape[0],), noise_scheduler.num_steps-1, device=device)
            noisy_images, noise = noise_scheduler.add_noise(images, t)

            predicted_noise = model(noisy_images, t)
            loss = criterion(noise, predicted_noise)
            loss_sum += loss.item()
            num_batches += 1
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {loss_sum/num_batches:.4f}")
        return loss_sum / len(dataloader)


def train_step(model, dataloader, noise_scheduler, criterion, optimizer, epoch, num_epochs, device):
    """训练步骤,计算训练集上的损失并更新模型参数"""
    # 设置模型为训练模式
    model.train()
    loss_sum = 0
    num_batches = 0
    pbar = tqdm(dataloader)
    for batch in pbar:
        # 获取一个batch的图像数据并移至指定设备
        images, _ = batch
        images = images.to(device)
        
        # 随机采样时间步t
        t = torch.randint(0, noise_scheduler.num_steps, (images.shape[0],), device=device)
        
        # 对图像添加噪声,获得带噪声的图像和噪声
        noisy_images, noise = noise_scheduler.add_noise(images, t)

        # 使用模型预测噪声
        predicted_noise = model(noisy_images, t)
        
        # 计算预测噪声和真实噪声之间的MSE损失
        loss = criterion(noise, predicted_noise)
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪,防止梯度爆炸
        optimizer.step()  # 更新参数

        # 累计损失并更新进度条
        loss_sum += loss.item()
        num_batches += 1
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss_sum/num_batches:.4f}")
        
    # 返回平均损失
    return loss_sum / len(dataloader)


def train(model, train_loader, test_loader, noise_scheduler, criterion, optimizer, device, num_epochs=100, img_size=32):
    """训练模型"""
    for epoch in range(num_epochs):
        train_loss = train_step(model, train_loader, noise_scheduler, criterion, optimizer, epoch, num_epochs, device)
        test_loss = test_step(model, test_loader, noise_scheduler, criterion, epoch, num_epochs, device)
        if epoch % 1 == 0:  #10
            # 采样10张图像
            images = sample(model, noise_scheduler, 10, (3, img_size, img_size), device)
            # 将图像从[-1, 1]范围缩放到[0, 1]范围,以便可视化
            images = ((images + 1) / 2).detach().cpu()
            fig = plot(images)
            os.makedirs("samples", exist_ok=True)
            fig.savefig(f"samples/epoch_{epoch}.png")

            # 添加可视化中间去噪步骤
            visualize_denoising_steps(model, noise_scheduler, device, img_size, epoch)
    return model

def visualize_denoising_steps(model, scheduler, device, img_size, epoch, num_steps=10):
    """可视化去噪过程中的中间步骤"""
    model.eval()
    with torch.no_grad():
        # 从标准正态分布采样初始噪声
        x_t = torch.randn(1, 3, img_size, img_size).to(device)
        
        # 保存去噪过程中的中间图像
        steps = np.linspace(0, scheduler.num_steps-1, num_steps).astype(int)
        steps = sorted(steps, reverse=True)  # 从T到0排序
        images = []
        images.append(x_t.cpu())  # 添加初始噪声
        
        # 逐步去噪
        for i, t in enumerate(steps):
            if i == 0:  # 跳过初始噪声，已经添加过了
                continue
            # 构造时间步batch
            t_batch = torch.tensor([t] * 1).to(device)
            
            # 获取采样需要的系数
            sqrt_recip_alpha_bar = scheduler.get(scheduler.sqrt_recip_alphas_bar, t_batch, x_t.shape)
            sqrt_recipm1_alpha_bar = scheduler.get(scheduler.sqrt_recipm1_alphas_bar, t_batch, x_t.shape)
            posterior_mean_coef1 = scheduler.get(scheduler.posterior_mean_coef1, t_batch, x_t.shape)
            posterior_mean_coef2 = scheduler.get(scheduler.posterior_mean_coef2, t_batch, x_t.shape)
            
            # 预测噪声
            predicted_noise = model(x_t, t_batch)
            
            # 计算x_0的预测值
            _x_0 = sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * predicted_noise
            # 计算后验分布均值
            model_mean = posterior_mean_coef1 * _x_0 + posterior_mean_coef2 * x_t
            # 计算后验分布方差的对数值
            model_log_var = scheduler.get(torch.log(torch.cat([scheduler.posterior_var[1:2], scheduler.betas[1:]])), t_batch, x_t.shape)
            
            if t > 0:
                # t>0时从后验分布采样
                noise = torch.randn_like(x_t).to(device)
                x_t = model_mean + torch.exp(0.5 * model_log_var) * noise
            else:
                # t=0时直接使用均值作为生成结果
                x_t = model_mean
            
            # 保存当前步骤的图像
            images.append(x_t.cpu())
        
        # 将最终结果裁剪到[-1,1]范围
        images[-1] = torch.clamp(images[-1], -1.0, 1.0)
        
        # 可视化所有步骤
        fig = plt.figure(figsize=(15, 3))
        for i in range(len(images)):
            plt.subplot(1, len(images), i+1)
            plt.axis("off")
            img = ((images[i][0] + 1) / 2).permute(1, 2, 0).numpy()
            plt.imshow(np.clip(img, 0, 1))
            if i == 0:
                plt.title("噪声")
            elif i == len(images)-1:
                plt.title("生成图像")
            else:
                plt.title(f"步骤 {steps[i-1]}")
        plt.tight_layout()
        os.makedirs("denoising_steps", exist_ok=True)
        plt.savefig(f"denoising_steps/epoch_{epoch}.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8) #128
    parser.add_argument('--epochs', type=int, default=1) #200
    parser.add_argument('--lr', type=float, default=4e-4) #1e-4
    parser.add_argument('--img_size', type=int, default=16)#32
    parser.add_argument('--subset_size', type=int, default=1000)  # 添加子集大小参数
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 使用子集大小参数
    train_loader, test_loader = load_transformed_dataset(args.img_size, args.batch_size, args.subset_size)
    noise_scheduler = NoiseScheduler(num_steps=500).to(device) #减少扩散步数
    model = SimpleUnet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    model = train(model, train_loader, test_loader, noise_scheduler, criterion, optimizer, device, args.epochs, args.img_size)
    torch.save(model.state_dict(), f"simple-unet-ddpm-{args.img_size}.pth")
