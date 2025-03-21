"""
配置管理工具
"""
import yaml
from typing import Dict, Any
import os

class ConfigManager:
    """
    配置管理类，用于加载和管理应用配置
    """
    
    @staticmethod
    def load_config(config_path: str = None) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
            
        Returns:
            配置字典
        """
        if config_path is None or not os.path.exists(config_path):
            # 返回默认配置
            return {
                'data_processing': {
                    'lookback': 20,
                    'test_size': 0.2,
                    'feature_selection': {
                        'correlation_threshold': 0.7,
                        'vif_threshold': 10,
                        'use_f_test': True
                    }
                },
                'models': {
                    'single_lstm': {
                        'hidden_dim': 64,
                        'dropout': 0.2,
                        'weight_decay': 1e-5
                    },
                    'dual_lstm': {
                        'hidden_dim': 32,
                        'num_layers': 2,
                        'dropout': 0.3,
                        'weight_decay': 1e-5
                    },
                    'arima': {
                        'max_p': 3,
                        'max_d': 2,
                        'max_q': 3,
                        'criterion': 'bic'
                    }
                },
                'training': {
                    'num_epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'num_runs': 30,
                    'early_stopping': {
                        'enabled': True,
                        'patience': 15,
                        'min_delta': 0.0001
                    }
                }
            }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")
            return None
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> bool:
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            config_path: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            return True
        except Exception as e:
            print(f"保存配置文件失败: {str(e)}")
            return False
    
    @staticmethod
    def get_model_config(model_type: str) -> Dict[str, Any]:
        """
        获取指定模型的配置
        
        Args:
            model_type: 模型类型（'single_lstm', 'dual_lstm', 'arima'）
            
        Returns:
            模型配置字典
        """
        config = ConfigManager.load_config()
        return config['models'].get(model_type, {}) 