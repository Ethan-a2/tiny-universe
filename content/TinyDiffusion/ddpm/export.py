import torch
import argparse
from unet import SimpleUnet  # 确保路径正确
from diffusion import NoiseScheduler  # 确保路径正确

def export_onnx(model_path, output_path, img_size, num_steps):
    """将训练好的 PyTorch 模型导出为 ONNX 格式.

    Args:
        model_path (str): PyTorch 模型文件的路径.
        output_path (str): ONNX 模型文件的保存路径.
        img_size (int):  图像尺寸 (正方形).
        num_steps (int): Noise Scheduler的扩散步数.  用于创建dummy input.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载模型
    model = SimpleUnet().to(device)  # 初始化模型架构
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型权重
    model.eval()  # 设置为评估模式

    # 2. 创建 ONNX 导出需要的 dummy input
    #   - 输入1: noisy image (batch_size=1, channels=3, height=img_size, width=img_size)
    #   - 输入2: timestep (batch_size=1)
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)  # 噪声图像
    dummy_timesteps = torch.randint(0, num_steps, (1,), device=device).long()  # 时间步

    # 3. 导出 ONNX 模型
    torch.onnx.export(
        model,
        (dummy_input, dummy_timesteps),  # 模型输入 (tuple)
        output_path,  # ONNX 模型输出路径
        export_params=True,  # 包含模型参数
        opset_version=13,  # ONNX opset 版本 (根据你的环境选择)
        do_constant_folding=True,  # 优化 ONNX 图
        input_names=['noisy_image', 'timesteps'],  # 输入节点名称
        output_names=['predicted_noise'],  # 输出节点名称
        dynamic_axes={
            'noisy_image': {0: 'batch_size'},  # 动态 batch size
            'timesteps': {0: 'batch_size'},    # 动态 batch size (即使这里batch_size=1, 还是声明一下)
            'predicted_noise': {0: 'batch_size'} # 动态 batch size
        }
    )

    print(f"模型已导出到 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch DDPM model to ONNX format.")
    parser.add_argument("--model_path", type=str, default="simple-unet-ddpm-16.pth", help="Path to the PyTorch model (.pth) file.")
    parser.add_argument("--output_path", type=str, default="simple-unet-ddpm-16.onnx", help="Path to save the ONNX model (.onnx).")
    parser.add_argument("--img_size", type=int, default=32, help="Image size (height and width).")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of diffusion steps in the noise scheduler.")
    args = parser.parse_args()

    export_onnx(args.model_path, args.output_path, args.img_size, args.num_steps)