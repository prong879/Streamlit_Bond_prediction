"""
配置管理模块 - 负责加载和管理配置
"""
import os
import yaml
import argparse
from typing import Dict, Any, Optional

class ConfigManager:
    """配置管理类，负责加载和管理配置"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            print(f"警告: 配置文件 {self.config_path} 不存在，将使用默认配置")
            return self._get_default_config()
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"错误: 配置文件格式错误 - {e}")
                return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """返回默认配置"""
        # 这里定义默认配置，与原代码中的默认值保持一致
        return {
            "basic": {
                "data_file": "data/rlData.csv",
                "output_dir": "output",
                "random_seed": 42
            },
            "data_processing": {
                "lookback": 20,
                "test_size": 0.2,
                "feature_selection": {
                    "correlation_threshold": 0.7,
                    "vif_threshold": 10,
                    "use_f_test": True
                }
            },
            "models": {
                "single_lstm": {
                    "hidden_dim": 64,
                    "dropout": 0.2,
                    "weight_decay": 1e-5
                },
                "dual_lstm": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "dropout": 0.3,
                    "weight_decay": 1e-5
                },
                "arima": {
                    "max_p": 3,
                    "max_d": 2,
                    "max_q": 3,
                    "criterion": "bic"
                }
            },
            "training": {
                "num_epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "num_runs": 30,
                "early_stopping": {
                    "enabled": True,
                    "patience": 15,
                    "min_delta": 0.0001
                }
            },
            "visualization": {
                "style": "seaborn-whitegrid",
                "dpi": 300,
                "figsize": [12, 8],
                "cmap": "viridis",
                "font_family": "Times New Roman"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置键，支持点号分隔的嵌套键，如 "models.single_lstm.hidden_dim"
            default: 如果键不存在，返回的默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """
        从命令行参数更新配置
        
        Args:
            args: 命令行参数
        """
        # 将命令行参数转换为字典
        args_dict = vars(args)
        
        # 更新基本配置
        if 'data_file' in args_dict and args_dict['data_file']:
            self.config['basic']['data_file'] = args_dict['data_file']
            
        if 'output_dir' in args_dict and args_dict['output_dir']:
            self.config['basic']['output_dir'] = args_dict['output_dir']
            
        # 更新训练配置
        if 'num_runs' in args_dict and args_dict['num_runs']:
            self.config['training']['num_runs'] = args_dict['num_runs']
            
        if 'epochs' in args_dict and args_dict['epochs']:
            self.config['training']['num_epochs'] = args_dict['epochs']
            
        # 可以根据需要添加更多参数更新
    
    def save(self, path: Optional[str] = None) -> None:
        """
        保存当前配置到文件
        
        Args:
            path: 保存路径，如果为None则使用当前配置路径
        """
        save_path = path or self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
        print(f"配置已保存到 {save_path}") 