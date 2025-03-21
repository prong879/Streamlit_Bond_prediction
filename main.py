"""
股票价格预测系统 - 主程序入口
使用LSTM模型预测股票价格
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
import random

# 导入自定义模块
from src.data_processing.data_loader import load_data, create_output_dir
from src.data_processing.feature_engineering import add_technical_indicators, prepare_data_for_model, split_data
from src.feature_selection.selector import select_features
from src.models.dual_lstm_model import DualLSTM, train_dual_lstm_model
from src.models.single_lstm_model import SingleLSTM, train_single_lstm_model
from src.models.arma_model import train_arma_model, predict_arma, evaluate_arma_model, plot_arma_results, find_best_arima_params
from src.utils.visualization import (
    set_custom_style, set_plot_font, plot_stock_price, plot_correlation_matrix,
    plot_selected_features_correlation, plot_training_visualization,
    plot_validation_results, plot_prediction_results, plot_feature_importance
)
from src.utils.evaluation import (
    calculate_metrics, inverse_transform_predictions, save_evaluation_results
)
from src.utils.model_comparison import (
    compare_models, prepare_model_comparison_data, add_arma_to_comparison_data
)
from src.utils.model_training import train_multiple_models, train_multiple_arima_models
from src.utils.config_manager import ConfigManager

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="股票价格预测系统")
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='配置文件路径')
    parser.add_argument('--data_file', type=str, 
                        help='数据文件路径，覆盖配置文件中的设置')
    parser.add_argument('--output_dir', type=str, 
                        help='输出目录，覆盖配置文件中的设置')
    parser.add_argument('--num_runs', type=int, 
                        help='每个模型训练的次数，覆盖配置文件中的设置')
    parser.add_argument('--epochs', type=int, 
                        help='每次训练的轮数，覆盖配置文件中的设置')
    return parser.parse_args()

def main():
    """主程序入口"""
    # 解析命令行参数
    args = parse_args()
    
    # 初始化配置管理器
    config = ConfigManager(args.config)
    
    # 用命令行参数更新配置
    config.update_from_args(args)
    
    # 获取配置值
    data_file = config.get('basic.data_file')
    output_dir_base = config.get('basic.output_dir')
    random_seed = config.get('basic.random_seed', 42)
    lookback = config.get('data_processing.lookback', 20)
    
    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # 设置自定义样式
    try:
        # 设置Times New Roman字体
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'stix'
        print("成功设置Times New Roman字体")
    except Exception as e:
        print(f"设置字体时出错: {e}")
        print("将使用默认字体")

    # 设置自定义样式
    custom_cmap = set_custom_style()

    # 1. 加载数据
    data = load_data(data_file)
    print(data.head())
    print(data.shape)

    # 创建输出目录
    output_dir_base = create_output_dir(output_dir_base)
    output_dir_single = os.path.join(output_dir_base, 'single_lstm')
    output_dir_dual = os.path.join(output_dir_base, 'dual_lstm')
    output_dir_arma = os.path.join(output_dir_base, 'arma')
    output_dir_comparison = os.path.join(output_dir_base, 'comparison')
    
    # 确保所有子目录存在
    for directory in [output_dir_single, output_dir_dual, output_dir_arma, output_dir_comparison]:
        os.makedirs(directory, exist_ok=True)
    
    # 保存当前配置到输出目录，用于实验记录
    config.save(os.path.join(output_dir_base, 'config_used.yaml'))
    
    print("将同时训练单层LSTM、双层LSTM和ARMA模型并进行对比")

    # 2. 绘制原始股票价格走势图
    plot_stock_price(data, output_dir_base)

    # 3. 特征工程 - 增强版
    try:
        price_features = add_technical_indicators(data)
        print("成功添加技术指标，特征数量:", price_features.shape[1])
        print(price_features.columns.tolist())
    except Exception as e:
        print("添加技术指标时出错:", str(e))
        # 如果特征工程失败，回退到只使用收盘价
        price_features = data[['Close']].copy()

    # 4. 计算相关性矩阵
    correlation_matrix = price_features.corr()
    plot_correlation_matrix(correlation_matrix, output_dir_base, custom_cmap)

    # 5. 特征选择
    price_features_selected, correlation_matrix, selected_features = select_features(price_features, output_dir_base)
    
    # 6. 绘制选定特征的相关性矩阵
    plot_selected_features_correlation(price_features_selected, output_dir_base, custom_cmap)

    # 7. 数据归一化
    price_features_scaled, scaler = prepare_data_for_model(price_features_selected)
    print(price_features_scaled.shape)

    # 8. 数据集制作
    x_train, y_train, x_test, y_test = split_data(price_features_scaled, lookback)
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)

    # 9. 转换为PyTorch张量
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    # 10. 模型参数设置
    input_dim = x_train.shape[2]  # 动态获取特征维度
    print(f"模型输入维度: {input_dim}")
    output_dim = 1   # 预测后一天的收盘价
    num_epochs = config.get('training.num_epochs', 100)

    # 11. 训练单层LSTM模型
    hidden_dim_single = config.get('models.single_lstm.hidden_dim', 64)  # 单层LSTM使用更大的隐藏层
    print("\n训练单层LSTM模型...")
    model_single, hist_single, training_time_single, best_loss_single, single_stats = train_multiple_models(
        SingleLSTM,
        train_single_lstm_model,
        x_train_tensor, 
        y_train_tensor, 
        input_dim, 
        hidden_dim_single, 
        output_dim,
        num_runs=config.get('training.num_runs', 30),  # 从配置文件读取训练次数
        num_epochs=num_epochs,
        output_dir=output_dir_single,
        model_name="single_lstm"
    )
    
    # 12. 训练双层LSTM模型
    hidden_dim_dual = config.get('models.dual_lstm.hidden_dim', 32)  # 双层LSTM使用较小的隐藏层
    num_layers = config.get('models.dual_lstm.num_layers', 2)
    print("\n训练双层LSTM模型...")
    model_dual, hist_dual, training_time_dual, best_loss_dual, dual_stats = train_multiple_models(
        DualLSTM,
        train_dual_lstm_model,
        x_train_tensor, 
        y_train_tensor, 
        input_dim, 
        hidden_dim_dual, 
        output_dim,
        num_layers=num_layers,
        num_runs=config.get('training.num_runs', 30),  # 从配置文件读取训练次数
        num_epochs=num_epochs,
        output_dir=output_dir_dual,
        model_name="dual_lstm"
    )

    # 13. 单层LSTM模型预测 - 训练集
    model_single.eval()
    with torch.no_grad():
        y_train_pred_single = model_single(x_train_tensor)
    
    # 14. 双层LSTM模型预测 - 训练集
    model_dual.eval()
    with torch.no_grad():
        y_train_pred_dual = model_dual(x_train_tensor)
    
    # 15. 反归一化预测结果 - 单层LSTM
    train_predict_single, y_train_orig = inverse_transform_predictions(
        y_train_pred_single, y_train, scaler, selected_features
    )
    
    # 16. 反归一化预测结果 - 双层LSTM
    train_predict_dual, _ = inverse_transform_predictions(
        y_train_pred_dual, y_train, scaler, selected_features
    )
    
    # 17. 绘制训练可视化图 - 单层LSTM
    original = pd.DataFrame(y_train_orig)
    predict_single = pd.DataFrame(train_predict_single)
    plot_training_visualization(original, predict_single, hist_single, output_dir_single, title="Single LSTM Training Results")

    # 18. 绘制训练可视化图 - 双层LSTM
    predict_dual = pd.DataFrame(train_predict_dual)
    plot_training_visualization(original, predict_dual, hist_dual, output_dir_dual, title="Dual LSTM Training Results")

    # 19. 准备模型对比数据
    models_data = prepare_model_comparison_data(
        model_single, model_dual,
        x_test_tensor, y_test,
        scaler, selected_features,
        training_time_single, training_time_dual,
        calculate_metrics, inverse_transform_predictions
    )
    
    # 20. 获取单层LSTM模型的测试集预测结果和评估指标
    test_predict_orig_single = models_data['single_lstm']['test_predict']
    y_test_orig = models_data['single_lstm']['y_test']
    test_score_single = models_data['single_lstm']['rmse']
    test_direction_acc_single = models_data['single_lstm']['direction_acc']
    
    # 21. 获取双层LSTM模型的测试集预测结果和评估指标
    test_predict_orig_dual = models_data['dual_lstm']['test_predict']
    test_score_dual = models_data['dual_lstm']['rmse']
    test_direction_acc_dual = models_data['dual_lstm']['direction_acc']
    
    # 22. 计算训练集评估指标 - 单层LSTM
    print("\n计算单层LSTM训练集方向准确率:")
    train_score_single, train_direction_acc_single, train_std_single = calculate_metrics(y_train_orig, train_predict_single)
    
    # 23. 计算训练集评估指标 - 双层LSTM
    print("\n计算双层LSTM训练集方向准确率:")
    train_score_dual, train_direction_acc_dual, train_std_dual = calculate_metrics(y_train_orig, train_predict_dual)
    
    # 24. 计算测试集标准差
    _, _, test_std_single = calculate_metrics(y_test_orig, test_predict_orig_single)
    _, _, test_std_dual = calculate_metrics(y_test_orig, test_predict_orig_dual)
    
    # 25. 打印评估结果
    print("\n单层LSTM模型评估结果:")
    print(f'训练集 RMSE: {train_score_single:.4f}')
    print(f'测试集 RMSE: {test_score_single:.4f}')
    print(f'训练集方向准确率: {train_direction_acc_single:.2f}%')
    print(f'测试集方向准确率: {test_direction_acc_single:.2f}%')
    
    print("\n双层LSTM模型评估结果:")
    print(f'训练集 RMSE: {train_score_dual:.4f}')
    print(f'测试集 RMSE: {test_score_dual:.4f}')
    print(f'训练集方向准确率: {train_direction_acc_dual:.2f}%')
    print(f'测试集方向准确率: {test_direction_acc_dual:.2f}%')
    
    # 26. 保存评估结果 - 单层LSTM
    save_evaluation_results(
        output_dir_single, train_score_single, test_score_single, training_time_single, 
        train_direction_acc_single, test_direction_acc_single
    )
    
    # 27. 保存评估结果 - 双层LSTM
    save_evaluation_results(
        output_dir_dual, train_score_dual, test_score_dual, training_time_dual, 
        train_direction_acc_dual, test_direction_acc_dual
    )
    
    # 28. 绘制验证结果图 - 单层LSTM
    plot_validation_results(
        price_features_selected, y_train_orig, train_predict_single, 
        y_test_orig, test_predict_orig_single, train_score_single, test_score_single, 
        train_std_single, test_std_single, lookback, output_dir_single,
        title="Single LSTM Validation Results"
    )
    
    # 29. 绘制验证结果图 - 双层LSTM
    plot_validation_results(
        price_features_selected, y_train_orig, train_predict_dual, 
        y_test_orig, test_predict_orig_dual, train_score_dual, test_score_dual, 
        train_std_dual, test_std_dual, lookback, output_dir_dual,
        title="Dual LSTM Validation Results"
    )
    
    # 30. 绘制预测结果图 - 单层LSTM
    plot_prediction_results(
        price_features_selected, train_predict_single, test_predict_orig_single, 
        train_std_single, test_std_single, lookback, train_score_single, test_score_single, 
        output_dir_single, title="Single LSTM Prediction Results"
    )
    
    # 31. 绘制预测结果图 - 双层LSTM
    plot_prediction_results(
        price_features_selected, train_predict_dual, test_predict_orig_dual, 
        train_std_dual, test_std_dual, lookback, train_score_dual, test_score_dual, 
        output_dir_dual, title="Dual LSTM Prediction Results"
    )
    
    # 32. 模型对比分析
    compare_models(models_data, output_dir_comparison)
    
    # 33. 特征重要性分析
    try:
        plot_feature_importance(correlation_matrix, output_dir_base)
    except Exception as e:
        print("特征重要性分析失败:", str(e))
        
    # 34. 训练ARMA模型
    print("\nTraining ARMA model...")
    # 首先寻找最优ARIMA参数
    print("Finding optimal ARIMA parameters...")
    
    # 使用多次训练模块进行ARIMA模型训练
    best_model, best_order, arma_training_time, arima_stats = train_multiple_arima_models(
        find_best_arima_params,
        y_train_orig,
        num_runs=config.get('training.num_runs', 10) // 3,  # ARIMA训练次数减少，因为较慢
        max_p=config.get('models.arima.max_p', 3),
        max_d=config.get('models.arima.max_d', 2),
        max_q=config.get('models.arima.max_q', 3),
        criterion=config.get('models.arima.criterion', 'bic'),
        output_dir=output_dir_arma
    )
    
    if best_model is not None:
        print(f"Using optimal ARIMA model with parameters {best_order}")
        arma_model = best_model
        # 记录训练时间和参数信息
        with open(os.path.join(output_dir_arma, 'arima_training_info.txt'), 'w') as f:
            f.write(f"Best ARIMA Order: {best_order}\n")
            f.write(f"Best BIC: {best_model.bic:.4f}\n")
            f.write(f"Best AIC: {best_model.aic:.4f}\n")
            f.write(f"Search Time: {arma_training_time:.2f} seconds\n")
    else:
        # 如果参数搜索失败，使用默认参数
        print("Parameter search failed, using default parameters (0, 1, 1)")
        arma_order = (0, 1, 1)  # 默认参数 (p, d, q)
        arma_model, arma_training_time = train_arma_model(
            y_train_orig, order=arma_order, output_dir=output_dir_arma
        )
    
    # 35. ARMA模型预测 - 训练集
    print("进行ARMA模型训练集预测...")
    try:
        arma_train_pred = predict_arma(
            arma_model, 
            len(y_train_orig), 
            y_train=y_train_orig, 
            is_train_set=True  # 指定为训练集预测
        )
        print(f"训练集预测完成，预测结果长度: {len(arma_train_pred)}")
    except Exception as e:
        print(f"训练集预测失败: {str(e)}")
        # 如果预测失败，创建一个与训练集等长的零数组
        arma_train_pred = np.zeros(len(y_train_orig))
    
    # 36. ARMA模型预测 - 测试集
    print("进行ARMA模型测试集预测...")
    try:
        # 确保y_train_orig是numpy数组
        if isinstance(y_train_orig, pd.DataFrame):
            y_train_np = y_train_orig.values
        else:
            y_train_np = np.array(y_train_orig)
            
        arma_test_pred = predict_arma(
            arma_model, 
            len(y_test_orig), 
            y_train=y_train_np, 
            is_train_set=False  # 指定为测试集预测
        )
        print(f"测试集预测完成，预测结果长度: {len(arma_test_pred)}")
        
        # 打印ARMA测试集预测结果的前几个值，用于调试
        print("ARMA测试集预测结果前5个值:", arma_test_pred[:5])
    except Exception as e:
        print(f"测试集预测失败: {str(e)}")
        # 如果预测失败，创建一个与测试集等长的零数组
        arma_test_pred = np.zeros(len(y_test_orig))
    
    # 确保测试集预测结果的长度与测试集一致
    if isinstance(arma_test_pred, pd.Series):
        if len(arma_test_pred) > len(y_test_orig):
            print(f"截断ARMA测试集预测结果从{len(arma_test_pred)}到{len(y_test_orig)}")
            arma_test_pred = arma_test_pred.iloc[:len(y_test_orig)]
        elif len(arma_test_pred) < len(y_test_orig):
            print(f"ARMA测试集预测结果长度({len(arma_test_pred)})小于测试集长度({len(y_test_orig)})")
            # 创建一个与测试集等长的Series，用预测值填充前部分，用最后一个预测值填充后部分
            full_pred = pd.Series(index=range(len(y_train_orig), len(y_train_orig) + len(y_test_orig)))
            for i in range(len(arma_test_pred)):
                full_pred.iloc[i] = arma_test_pred.iloc[i]
            # 用最后一个预测值填充剩余部分
            if len(arma_test_pred) > 0:
                last_value = arma_test_pred.iloc[-1]
                for i in range(len(arma_test_pred), len(y_test_orig)):
                    full_pred.iloc[i] = last_value
            arma_test_pred = full_pred
    else:
        arma_test_pred_array = np.array(arma_test_pred).flatten()
        if len(arma_test_pred_array) > len(y_test_orig):
            arma_test_pred = arma_test_pred_array[:len(y_test_orig)]
        elif len(arma_test_pred_array) < len(y_test_orig):
            # 创建一个与测试集等长的数组，用预测值填充前部分，用最后一个预测值填充后部分
            full_pred = np.zeros(len(y_test_orig))
            full_pred[:len(arma_test_pred_array)] = arma_test_pred_array
            if len(arma_test_pred_array) > 0:
                full_pred[len(arma_test_pred_array):] = arma_test_pred_array[-1]
            arma_test_pred = full_pred
    
    # 37. 评估ARMA模型 - 训练集
    arma_train_score, arma_train_direction_acc, arma_train_std = evaluate_arma_model(
        y_train_orig, arma_train_pred, start_point=10  # 跳过前10个点进行评估
    )
    
    # 38. 评估ARMA模型 - 测试集
    arma_test_score, arma_test_direction_acc, arma_test_std = evaluate_arma_model(
        y_test_orig, arma_test_pred, start_point=0  # 测试集不需要跳过
    )
    
    # 添加调试信息，比较不同模型的预测结果
    print("\n预测结果统计特性对比:")
    print("真实值统计:")
    print(f"  - 均值: {np.mean(y_test_orig):.4f}")
    print(f"  - 标准差: {np.std(y_test_orig):.4f}")
    print(f"  - 最小值: {np.min(y_test_orig):.4f}")
    print(f"  - 最大值: {np.max(y_test_orig):.4f}")
    
    print("\n单层LSTM预测统计:")
    print(f"  - 均值: {np.mean(test_predict_orig_single):.4f}")
    print(f"  - 标准差: {np.std(test_predict_orig_single):.4f}")
    print(f"  - 最小值: {np.min(test_predict_orig_single):.4f}")
    print(f"  - 最大值: {np.max(test_predict_orig_single):.4f}")
    
    print("\n双层LSTM预测统计:")
    print(f"  - 均值: {np.mean(test_predict_orig_dual):.4f}")
    print(f"  - 标准差: {np.std(test_predict_orig_dual):.4f}")
    print(f"  - 最小值: {np.min(test_predict_orig_dual):.4f}")
    print(f"  - 最大值: {np.max(test_predict_orig_dual):.4f}")
    
    print("\nARMA预测统计:")
    if isinstance(arma_test_pred, pd.Series):
        arma_test_pred_array = arma_test_pred.values
    else:
        arma_test_pred_array = np.array(arma_test_pred).reshape(-1, 1)
    print(f"  - 均值: {np.mean(arma_test_pred_array):.4f}")
    print(f"  - 标准差: {np.std(arma_test_pred_array):.4f}")
    print(f"  - 最小值: {np.min(arma_test_pred_array):.4f}")
    print(f"  - 最大值: {np.max(arma_test_pred_array):.4f}")
    
    # 检查预测结果的差异
    print("\n预测结果差异:")
    print(f"单层LSTM vs 双层LSTM: {np.mean(np.abs(test_predict_orig_single - test_predict_orig_dual)):.4f}")
    print(f"单层LSTM vs ARMA: {np.mean(np.abs(test_predict_orig_single - arma_test_pred_array)):.4f}")
    print(f"双层LSTM vs ARMA: {np.mean(np.abs(test_predict_orig_dual - arma_test_pred_array)):.4f}")
    
    # 检查方向预测的差异
    single_direction = np.sign(np.diff(test_predict_orig_single.flatten()))
    dual_direction = np.sign(np.diff(test_predict_orig_dual.flatten()))
    arma_direction = np.sign(np.diff(arma_test_pred_array.flatten()))
    
    print("\n方向预测一致性:")
    print(f"单层LSTM vs 双层LSTM: {np.mean(single_direction == dual_direction) * 100:.2f}%")
    print(f"单层LSTM vs ARMA: {np.mean(single_direction == arma_direction) * 100:.2f}%")
    print(f"双层LSTM vs ARMA: {np.mean(dual_direction == arma_direction) * 100:.2f}%")
    
    # 39. 打印ARMA模型评估结果
    print("\nARMA模型评估结果:")
    print(f'训练集 RMSE: {arma_train_score:.4f}')
    print(f'测试集 RMSE: {arma_test_score:.4f}')
    print(f'训练集方向准确率: {arma_train_direction_acc:.2f}%')
    print(f'测试集方向准确率: {arma_test_direction_acc:.2f}%')
    
    # 40. 保存ARMA模型评估结果
    with open(os.path.join(output_dir_arma, 'arma_evaluation.txt'), 'w') as f:
        f.write(f'Training Set RMSE: {arma_train_score:.4f}\n')
        f.write(f'Test Set RMSE: {arma_test_score:.4f}\n')
        f.write(f'Training Time: {arma_training_time:.4f} seconds\n')
        f.write(f'Training Set Direction Accuracy: {arma_train_direction_acc:.2f}%\n')
        f.write(f'Test Set Direction Accuracy: {arma_test_direction_acc:.2f}%\n')
    
    # 41. 绘制ARMA模型预测结果
    plot_arma_results(
        y_train_orig.flatten(), y_test_orig.flatten(),
        arma_train_pred, arma_test_pred,
        arma_train_score, arma_test_score,
        output_dir_arma
    )
    
    # 42. 将ARMA模型结果添加到模型对比数据中
    # 确保ARMA预测结果的格式与LSTM模型一致
    if isinstance(arma_test_pred, pd.Series):
        arma_test_pred_array = arma_test_pred.values.reshape(-1, 1)
    else:
        arma_test_pred_array = np.array(arma_test_pred).reshape(-1, 1)
    
    models_data = add_arma_to_comparison_data(
        models_data, 
        arma_test_pred_array, 
        arma_test_score, 
        arma_test_direction_acc, 
        arma_training_time
    )
    
    # 43. 重新生成包含ARMA模型的对比图
    compare_models(models_data, output_dir_comparison)
    
    # 44. 比较LSTM和ARMA模型
    print("\nLSTM vs ARMA模型对比:")
    print(f'单层LSTM测试集RMSE: {test_score_single:.4f}, 方向准确率: {test_direction_acc_single:.2f}%')
    print(f'双层LSTM测试集RMSE: {test_score_dual:.4f}, 方向准确率: {test_direction_acc_dual:.2f}%')
    print(f'ARMA测试集RMSE: {arma_test_score:.4f}, 方向准确率: {arma_test_direction_acc:.2f}%')
    
    # 45. 保存模型对比结果
    with open(os.path.join(output_dir_comparison, 'model_comparison_with_arma.txt'), 'w') as f:
        f.write("LSTM vs ARMA Model Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Single LSTM Model:\n")
        f.write(f"  - Test RMSE: {test_score_single:.4f}\n")
        f.write(f"  - Test Direction Accuracy: {test_direction_acc_single:.2f}%\n")
        f.write(f"  - Training Time: {training_time_single:.2f} seconds\n\n")
        
        f.write("Dual LSTM Model:\n")
        f.write(f"  - Test RMSE: {test_score_dual:.4f}\n")
        f.write(f"  - Test Direction Accuracy: {test_direction_acc_dual:.2f}%\n")
        f.write(f"  - Training Time: {training_time_dual:.2f} seconds\n\n")
        
        f.write("ARMA Model:\n")
        f.write(f"  - Test RMSE: {arma_test_score:.4f}\n")
        f.write(f"  - Test Direction Accuracy: {arma_test_direction_acc:.2f}%\n")
        f.write(f"  - Training Time: {arma_training_time:.2f} seconds\n\n")
        
        # 确定最佳模型
        best_rmse = min(test_score_single, test_score_dual, arma_test_score)
        if best_rmse == test_score_single:
            best_model = "Single LSTM"
        elif best_rmse == test_score_dual:
            best_model = "Dual LSTM"
        else:
            best_model = "ARMA"
            
        best_dir_acc = max(test_direction_acc_single, test_direction_acc_dual, arma_test_direction_acc)
        if best_dir_acc == test_direction_acc_single:
            best_dir_model = "Single LSTM"
        elif best_dir_acc == test_direction_acc_dual:
            best_dir_model = "Dual LSTM"
        else:
            best_dir_model = "ARMA"
        
        f.write("Conclusion:\n")
        f.write(f"  - Best model for RMSE: {best_model} ({best_rmse:.4f})\n")
        f.write(f"  - Best model for Direction Accuracy: {best_dir_model} ({best_dir_acc:.2f}%)\n")
    
    # 46. 保存最佳模型信息到README文件
    with open(os.path.join(output_dir_comparison, 'README.md'), 'w') as f:
        f.write("# 模型对比结果\n\n")
        f.write("## 性能指标对比\n\n")
        f.write("| 模型 | RMSE | 方向准确率 | 训练时间(秒) |\n")
        f.write("|------|------|------------|-------------|\n")
        f.write(f"| 单层LSTM | {test_score_single:.4f} | {test_direction_acc_single:.2f}% | {training_time_single:.2f} |\n")
        f.write(f"| 双层LSTM | {test_score_dual:.4f} | {test_direction_acc_dual:.2f}% | {training_time_dual:.2f} |\n")
        f.write(f"| ARMA | {arma_test_score:.4f} | {arma_test_direction_acc:.2f}% | {arma_training_time:.2f} |\n\n")
        
        f.write("## 结论\n\n")
        f.write(f"- RMSE最低的模型: **{best_model}** ({best_rmse:.4f})\n")
        f.write(f"- 方向准确率最高的模型: **{best_dir_model}** ({best_dir_acc:.2f}%)\n\n")
        
        f.write("## 模型对比图\n\n")
        f.write("![模型对比图](model_comparison.png)\n")
    
    print("\n所有模型训练和评估完成，结果已保存到输出目录。")

if __name__ == "__main__":
    main()
