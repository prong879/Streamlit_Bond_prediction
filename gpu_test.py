import torch
import sys
import os

# 添加src目录到路径，以便能够导入models模块
sys.path.insert(0, os.path.abspath('src'))

from src.models.lstm_model import get_device, LSTMModel

def test_gpu_support():
    """
    测试GPU支持情况
    """
    print("="*50)
    print("PyTorch GPU支持测试")
    print("="*50)
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA是否可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        
        # 获取GPU设备
        device = get_device()
        print(f"使用设备: {device}")
        
        # 创建一个小的LSTM模型
        model = LSTMModel(
            input_dim=5,
            hidden_dim=10,
            num_layers=2,
            output_dim=1
        ).to(device)
        print(f"模型位于设备: {next(model.parameters()).device}")
        
        # 创建随机输入数据
        x = torch.randn(32, 10, 5).to(device)  # batch_size=32, sequence_length=10, features=5
        
        # 前向传播
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        # 预热
        for _ in range(5):
            _ = model(x)
        
        # 计时
        start_time.record()
        for _ in range(100):
            _ = model(x)
        end_time.record()
        
        # 等待所有GPU操作完成
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"100次前向传播耗时: {elapsed_time:.2f} ms")
        print(f"平均每次耗时: {elapsed_time/100:.2f} ms")
        
        # 测试CPU性能
        model_cpu = model.cpu()
        x_cpu = x.cpu()
        
        # 预热CPU
        for _ in range(5):
            _ = model_cpu(x_cpu)
        
        # 测量CPU时间
        import time
        cpu_start = time.time()
        for _ in range(100):
            _ = model_cpu(x_cpu)
        cpu_end = time.time()
        
        cpu_elapsed = (cpu_end - cpu_start) * 1000  # 转换为毫秒
        print(f"CPU上100次前向传播耗时: {cpu_elapsed:.2f} ms")
        print(f"CPU上平均每次耗时: {cpu_elapsed/100:.2f} ms")
        
        if elapsed_time < cpu_elapsed:
            speedup = cpu_elapsed / elapsed_time
            print(f"GPU加速比: {speedup:.2f}x")
            print("GPU加速正常工作!")
        else:
            print("警告: GPU没有比CPU快，请检查配置")
    else:
        print("未检测到CUDA支持，PyTorch将在CPU上运行")

if __name__ == "__main__":
    test_gpu_support() 