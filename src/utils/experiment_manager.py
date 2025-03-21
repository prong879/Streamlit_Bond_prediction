"""
实验管理模块 - 负责管理不同实验的配置和结果
"""
import os
import time
import shutil
from typing import Dict, Any, Optional
from src.utils.config_manager import ConfigManager

class ExperimentManager:
    """实验管理类，负责管理不同实验的配置和结果"""
    
    def __init__(self, base_dir: str = "experiments"):
        """
        初始化实验管理器
        
        Args:
            base_dir: 实验基础目录
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def create_experiment(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> str:
        """
        创建新实验
        
        Args:
            name: 实验名称，如果为None则使用时间戳
            config: 实验配置，如果为None则使用默认配置
            
        Returns:
            实验目录路径
        """
        # 生成实验名称
        if name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            name = f"experiment_{timestamp}"
            
        # 创建实验目录
        exp_dir = os.path.join(self.base_dir, name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # 保存配置
        if config is not None:
            config_manager = ConfigManager()
            config_manager.config = config
            config_manager.save(os.path.join(exp_dir, 'config.yaml'))
            
        return exp_dir
    
    def load_experiment(self, name: str) -> ConfigManager:
        """
        加载已有实验
        
        Args:
            name: 实验名称
            
        Returns:
            配置管理器
        """
        exp_dir = os.path.join(self.base_dir, name)
        config_path = os.path.join(exp_dir, 'config.yaml')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"实验 {name} 的配置文件不存在")
            
        return ConfigManager(config_path)
    
    def list_experiments(self) -> Dict[str, str]:
        """
        列出所有实验
        
        Returns:
            实验名称到路径的映射
        """
        experiments = {}
        
        for name in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'config.yaml')):
                experiments[name] = path
                
        return experiments
    
    def copy_experiment(self, source: str, target: str) -> str:
        """
        复制实验
        
        Args:
            source: 源实验名称
            target: 目标实验名称
            
        Returns:
            目标实验目录路径
        """
        source_dir = os.path.join(self.base_dir, source)
        target_dir = os.path.join(self.base_dir, target)
        
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"源实验 {source} 不存在")
            
        if os.path.exists(target_dir):
            raise FileExistsError(f"目标实验 {target} 已存在")
            
        # 只复制配置文件
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(
            os.path.join(source_dir, 'config.yaml'),
            os.path.join(target_dir, 'config.yaml')
        )
        
        return target_dir
    
    def compare_experiments(self, experiment_names: list) -> Dict[str, Any]:
        """
        比较多个实验的配置
        
        Args:
            experiment_names: 实验名称列表
            
        Returns:
            配置差异字典
        """
        if len(experiment_names) < 2:
            raise ValueError("至少需要两个实验进行比较")
            
        configs = {}
        for name in experiment_names:
            config_manager = self.load_experiment(name)
            configs[name] = config_manager.config
            
        # 找出所有配置键
        all_keys = set()
        for config in configs.values():
            self._collect_keys(config, "", all_keys)
            
        # 比较配置差异
        differences = {}
        for key in sorted(all_keys):
            values = {}
            for name, config in configs.items():
                value = self._get_nested_value(config, key)
                values[name] = value
                
            # 检查是否有差异
            if len(set(str(v) for v in values.values())) > 1:
                differences[key] = values
                
        return differences
    
    def _collect_keys(self, config: Dict[str, Any], prefix: str, keys: set):
        """收集配置中的所有键"""
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._collect_keys(value, full_key, keys)
            else:
                keys.add(full_key)
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """获取嵌套配置值"""
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
                
        return value 