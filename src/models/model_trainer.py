# src/models/model_trainer.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import json
import os
from datetime import datetime, timedelta

from data.data_collector import DataCollector
from data.feature_engineer import FeatureEngineer
from utils.config_loader import ConfigLoader
from utils.logger import get_logger, performance_logger, debug_logger

class ModelTrainer:
    """
    A股模型训练器 - 适配新版DataCollector接口
    """
    
    def __init__(self, config_path="config"):
        config_loader = ConfigLoader(config_path)
        self.config = config_loader.load_all_configs()
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer(self.config['features'])
        self.model = None
        self.feature_columns = []
        self.models_loaded = False  # 添加模型加载状态标志
        
        # 初始化日志记录器
        self.logger = get_logger(__name__)
        self.performance_logger = performance_logger
        self.debug_logger = debug_logger
        
        self.logger.info("模型训练器初始化完成")
    
    def train_single_stock(self, symbol, period="2y", save_model=True, cache_check=True):
        """
        训练单只股票的模型 - 适配新版接口
        """
        self.logger.info(f"开始训练股票 {symbol} 的模型", extra={
            'trading_context': {
                'symbol': symbol,
                'period': period,
                'action': 'model_training_start'
            }
        })
        
        try:
            # 1. 数据收集 - 使用新的接口
            self.debug_logger.log_data_processing(symbol, "data_collection", {"period": period})
            raw_data = self._get_stock_data_by_period(symbol, period, cache_check=cache_check)
            
            if raw_data is None or raw_data.empty:
                self.logger.error(f"无法获取 {symbol} 的数据", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'error': 'data_fetch_failed'
                    }
                })
                return None
            
            # 2. 特征工程 - 修改目标变量类型
            self.debug_logger.log_data_processing(symbol, "feature_engineering", {"data_points": len(raw_data)})
            processed_data = self.feature_engineer.create_features(
                raw_data, target_type="close_close"  # 修改为close_close
            )
            
            if processed_data is None or 'target' not in processed_data.columns:
                self.logger.error(f"特征工程失败或缺少目标变量 {symbol}", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'error': 'feature_engineering_failed'
                    }
                })
                return None
            
            # 3. 准备训练数据
            X_train, X_test, y_train, y_test = self._prepare_training_data(processed_data)
            
            if len(X_train) == 0:
                self.logger.error(f"训练数据不足: {symbol}", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'error': 'insufficient_training_data',
                        'data_points': len(processed_data)
                    }
                })
                return None
            
            # 4. 清理数据（修复inf和过大值问题）
            self.debug_logger.log_data_processing(symbol, "data_cleaning", {
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            })
            X_train, X_test = self._clean_training_data(X_train, X_test)
            
            if len(X_train) == 0:
                self.logger.error(f"数据清理后训练数据为空: {symbol}", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'error': 'empty_data_after_cleaning'
                    }
                })
                return None
            
            # 5. 训练模型
            accuracy = self._train_xgboost(X_train, X_test, y_train, y_test, symbol)
            
            # 6. 保存模型
            if save_model and accuracy is not None:
                self._save_model(symbol, accuracy, self.feature_columns)
            
            # 记录训练结果
            if accuracy is not None:
                self.performance_logger.log_performance({
                    'symbol': symbol,
                    'accuracy': accuracy,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features_count': len(self.feature_columns),
                    'period': period,
                    'action': 'model_training_complete'
                })
            
            self.logger.info(f"股票 {symbol} 训练完成，准确率: {accuracy:.3f}", extra={
                'trading_context': {
                    'symbol': symbol,
                    'accuracy': accuracy,
                    'action': 'model_training_complete'
                }
            })
            return accuracy
            
        except Exception as e:
            error_msg = str(e) if e is not None else "未知错误"
            self.logger.error(f"训练股票 {symbol} 时出错: {error_msg}", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'training_exception',
                    'error_message': error_msg
                }
            })
            return None
    
    def _get_stock_data_by_period(self, symbol, period, cache_check=True):
        """
        根据period参数获取股票数据 - 适配新版DataCollector
        """
        try:
            # 将period转换为具体的日期范围
            end_date = datetime.now()
            
            if period == "2y":
                start_date = end_date - timedelta(days=730)  # 2年
            elif period == "1y":
                start_date = end_date - timedelta(days=365)  # 1年
            elif period == "6m":
                start_date = end_date - timedelta(days=180)  # 6个月
            else:
                # 默认2年
                start_date = end_date - timedelta(days=730)
            
            start_date_str = start_date.strftime("%Y%m%d")
            end_date_str = end_date.strftime("%Y%m%d")
            
            self.debug_logger.log_data_processing(symbol, "data_fetch", {
                "start_date": start_date_str,
                "end_date": end_date_str
            })
            
            data = self.data_collector.get_stock_data(
                symbol, start_date_str, end_date_str, adjust_type="qfq", source="auto", cache_check=cache_check
            )
            
            if data is None or data.empty:
                self.logger.warning(f"获取到空数据: {symbol}", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'warning': 'empty_data_received'
                    }
                })
                return None
                
            self.debug_logger.log_data_processing(symbol, "data_fetch_complete", {
                "data_points": len(data),
                "columns": list(data.columns)
            })
            
            return data
            
        except Exception as e:
            self.logger.error(f"获取股票数据失败 {symbol}: {e}", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'data_fetch_exception',
                    'error_message': str(e)
                }
            })
            return None
    
    def _clean_training_data(self, X_train, X_test):
        """
        清理训练数据中的inf和过大值 - 修复pandas警告
        """
        try:
            # 复制数据避免修改原始数据
            X_train_clean = X_train.copy()
            X_test_clean = X_test.copy()
            
            # 1. 替换inf和-inf为NaN
            X_train_clean = X_train_clean.replace([np.inf, -np.inf], np.nan)
            X_test_clean = X_test_clean.replace([np.inf, -np.inf], np.nan)
            
            # 2. 计算每列的统计量用于填充
            fill_values = {}
            clip_bounds = {}
            
            for col in X_train_clean.columns:
                # 计算中位数和四分位距
                median_val = X_train_clean[col].median()
                q1 = X_train_clean[col].quantile(0.25)
                q3 = X_train_clean[col].quantile(0.75)
                iqr = q3 - q1
                
                # 存储填充值和边界
                fill_values[col] = median_val
                
                # 定义异常值边界
                if iqr > 0:  # 避免除零
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    clip_bounds[col] = (lower_bound, upper_bound)
                else:
                    clip_bounds[col] = (None, None)
            
            # 修复：避免链式赋值，使用一次性填充
            # 填充NaN值（使用中位数）
            X_train_clean = X_train_clean.fillna(fill_values)
            X_test_clean = X_test_clean.fillna(fill_values)
            
            # 缩放过大的值到合理范围
            for col, (lower_bound, upper_bound) in clip_bounds.items():
                if lower_bound is not None and upper_bound is not None:
                    # 对训练集和测试集进行裁剪
                    X_train_clean[col] = np.clip(X_train_clean[col], lower_bound, upper_bound)
                    X_test_clean[col] = np.clip(X_test_clean[col], lower_bound, upper_bound)
            
            # 3. 标准化数据（可选，但有助于XGBoost训练）
            # 这里我们使用RobustScaler，对异常值不敏感
            scaler = RobustScaler()
            
            # 只在训练集上拟合，然后转换训练集和测试集
            X_train_scaled = scaler.fit_transform(X_train_clean)
            X_test_scaled = scaler.transform(X_test_clean)
            
            # 转换回DataFrame
            X_train_final = pd.DataFrame(X_train_scaled, columns=X_train_clean.columns, index=X_train_clean.index)
            X_test_final = pd.DataFrame(X_test_scaled, columns=X_test_clean.columns, index=X_test_clean.index)
            
            return X_train_final, X_test_final
            
        except Exception as e:
            self.logger.error(f"数据清理失败: {e}", extra={
                'trading_context': {
                    'error': 'data_cleaning_exception',
                    'error_message': str(e)
                }
            })
            # 如果清理失败，返回原始数据但替换inf
            X_train_safe = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_test_safe = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
            return X_train_safe, X_test_safe
    
    def train_multiple_stocks(self, symbols, period="2y", cache_check=True):
        """
        批量训练多只股票模型
        """
        self.logger.info(f"开始批量训练 {len(symbols)} 只股票", extra={
            'trading_context': {
                'action': 'batch_training_start',
                'stock_count': len(symbols),
                'period': period
            }
        })
        
        results = {}
        start_time = datetime.now()
        
        for symbol in symbols:
            accuracy = self.train_single_stock(symbol, period, cache_check=cache_check)
            results[symbol] = {
                'accuracy': accuracy,
                'status': 'SUCCESS' if accuracy is not None else 'FAILED'
            }
            
        # 生成训练报告
        training_report = self._generate_training_report(results)
        
        # 记录批量训练结果
        training_duration = (datetime.now() - start_time).total_seconds()
        self.performance_logger.log_performance({
            'batch_training_duration': training_duration,
            'total_stocks': len(symbols),
            'successful_trains': training_report['successful_trains'],
            'failed_trains': training_report['failed_trains'],
            'average_accuracy': training_report['average_accuracy']
        })
        
        return results
    
    def _prepare_training_data(self, data, test_size=0.2):
        """
        准备训练和测试数据
        """
        try:
            # 确保数据足够
            if len(data) < 100:
                self.logger.warning(f"数据量不足，只有 {len(data)} 条记录", extra={
                    'trading_context': {
                        'warning': 'insufficient_data',
                        'data_points': len(data)
                    }
                })
                return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
            
            # 确保目标变量存在且有效
            if 'target' not in data.columns:
                self.logger.error("缺少目标变量 'target'", extra={
                    'trading_context': {
                        'error': 'missing_target_variable'
                    }
                })
                return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
            
            # 检查目标变量是否只有单一类别
            unique_targets = data['target'].unique()
            if len(unique_targets) < 2:
                self.logger.warning(f"目标变量只有 {len(unique_targets)} 个类别，无法训练", extra={
                    'trading_context': {
                        'warning': 'single_class_target',
                        'unique_classes': len(unique_targets)
                    }
                })
                return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
            
            # 确保特征列存在
            if not self.feature_columns:
                self.feature_columns = [col for col in data.columns 
                                      if col not in ['target', 'ChangePct', 'Change', 
                                                   'Amplitude', 'Turnover', 'Amount'] 
                                      and pd.api.types.is_numeric_dtype(data[col])]
            
            # 检查是否有足够的特征
            if len(self.feature_columns) == 0:
                self.logger.error("没有找到有效的特征列", extra={
                    'trading_context': {
                        'error': 'no_valid_features'
                    }
                })
                return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
            
            X = data[self.feature_columns]
            y = data['target']
            
            # 时间序列分割
            split_idx = int(len(X) * (1 - test_size))
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            self.debug_logger.log_data_processing("training_data", "prepared", {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(self.feature_columns),
                "test_size": test_size
            })
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"准备训练数据失败: {e}", extra={
                'trading_context': {
                    'error': 'training_data_preparation_failed',
                    'error_message': str(e)
                }
            })
            return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test, symbol):
        """
        训练XGBoost模型
        """
        try:
            # 修复：确保模型正确初始化
            model_params = self.config.get('model', {})
            
            # 设置默认参数
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss',
                # 添加处理缺失值的参数
                'missing': np.nan,
                # 添加防止过拟合的参数
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }
            
            # 使用配置参数，如果没有则使用默认值
            params = {**default_params, **model_params}
            
            # 修复：创建新的模型实例
            model = xgb.XGBClassifier(**params)
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # 计算基准准确率（多数类）
            baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
            
            self.logger.info(f"模型准确率: {accuracy:.3f}, 基准准确率: {baseline_accuracy:.3f}", extra={
                'trading_context': {
                    'symbol': symbol,
                    'model_accuracy': accuracy,
                    'baseline_accuracy': baseline_accuracy
                }
            })
            
            # 特征重要性
            feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            self.logger.info(f"Top 5特征重要性: {top_features}", extra={
                'trading_context': {
                    'symbol': symbol,
                    'top_features': top_features
                }
            })
            
            # 修复：保存训练好的模型到实例变量
            self.model = model
            
            return accuracy
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'model_training_exception',
                    'error_message': str(e)
                }
            })
            # 修复：确保模型为None
            self.model = None
            return None
    
    def _save_model(self, symbol, accuracy, feature_columns):
        """
        保存训练好的模型
        """
        try:
            # 修复：检查模型是否存在
            if self.model is None:
                self.logger.error(f"无法保存模型，模型为None: {symbol}", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'error': 'model_save_failed_none'
                    }
                })
                return
            
            # 保存模型文件
            model_dir = f"models/xgboost/{symbol}"
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = f"{model_dir}/model.json"
            self.model.save_model(model_path)
            
            # 保存模型元数据 - 更新目标变量描述
            metadata = {
                'symbol': symbol,
                'model_type': 'XGBoost',
                'accuracy': accuracy,
                'feature_columns': feature_columns,
                'target': '明天收盘相对今天收盘的涨跌',  # 修改目标变量描述
                'training_date': datetime.now().isoformat(),
                'feature_count': len(feature_columns),
                'model_config': self.config.get('model', {})
            }
            
            metadata_path = f"{model_dir}/metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"模型已保存: {model_path}", extra={
                'trading_context': {
                    'symbol': symbol,
                    'model_path': model_path,
                    'accuracy': accuracy,
                    'action': 'model_saved'
                }
            })
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'model_save_exception',
                    'error_message': str(e)
                }
            })
    
    def _generate_training_report(self, results):
        """
        生成训练报告
        """
        try:
            successful_trains = [s for s, r in results.items() if r['status'] == 'SUCCESS']
            failed_trains = [s for s, r in results.items() if r['status'] == 'FAILED']
            
            # 修复：安全的准确率计算
            accuracies = []
            for r in results.values():
                if r.get('accuracy') is not None:
                    accuracies.append(r['accuracy'])
            
            avg_accuracy = np.mean(accuracies) if accuracies else 0
            max_accuracy = max(accuracies) if accuracies else 0
            min_accuracy = min(accuracies) if accuracies else 0
            
            report = {
                'total_stocks': len(results),
                'successful_trains': len(successful_trains),
                'failed_trains': len(failed_trains),
                'average_accuracy': float(avg_accuracy),  # 转换为float避免numpy类型
                'max_accuracy': float(max_accuracy),
                'min_accuracy': float(min_accuracy),
                'training_date': datetime.now().isoformat(),
                'successful_symbols': successful_trains,
                'failed_symbols': failed_trains,
                'details': results
            }
            
            # 保存报告
            report_dir = "results"
            os.makedirs(report_dir, exist_ok=True)
            report_path = f"{report_dir}/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"训练报告已保存: {report_path}", extra={
                'trading_context': {
                    'action': 'training_report_saved',
                    'report_path': report_path,
                    'successful_trains': len(successful_trains),
                    'failed_trains': len(failed_trains),
                    'average_accuracy': avg_accuracy
                }
            })
            return report
            
        except Exception as e:
            self.logger.error(f"生成训练报告失败: {e}", extra={
                'trading_context': {
                    'error': 'training_report_exception',
                    'error_message': str(e)
                }
            })
            return {}

    def train_online(self, X, y):
        """
        在线学习接口 - 供LearningModule调用
        """
        try:
            # 清理在线学习数据
            X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # 修复：模型初始化检查
            if self.model is None:
                self.logger.info("创建新的XGBoost模型进行在线学习", extra={
                    'trading_context': {
                        'action': 'online_learning_new_model'
                    }
                })
                model_params = self.config.get('model', {})
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'missing': np.nan
                }
                params = {**default_params, **model_params}
                
                self.model = xgb.XGBClassifier(**params)
            
            # 在线更新模型
            self.model.fit(X_clean, y, xgb_model=self.model.get_booster() if hasattr(self.model, 'get_booster') else None)
            
            # 评估性能
            y_pred = self.model.predict(X_clean)
            accuracy = accuracy_score(y, y_pred)
            
            self.logger.info(f"在线学习完成，准确率: {accuracy:.3f}", extra={
                'trading_context': {
                    'action': 'online_learning_complete',
                    'accuracy': accuracy,
                    'samples_trained': len(X)
                }
            })
            
            return {
                "accuracy": accuracy,
                "samples_trained": len(X)
            }
            
        except Exception as e:
            self.logger.error(f"在线学习失败: {e}", extra={
                'trading_context': {
                    'error': 'online_learning_exception',
                    'error_message': str(e)
                }
            })
            return {}
    
    def deploy_model(self):
        """
        部署模型 - 供LearningModule调用
        """
        try:
            if self.model is None:
                self.logger.warning("没有可部署的模型", extra={
                    'trading_context': {
                        'warning': 'no_model_to_deploy'
                    }
                })
                return False
                
            # 这里可以实现模型版本管理和部署逻辑
            self.logger.info("模型部署完成", extra={
                'trading_context': {
                    'action': 'model_deployed'
                }
            })
            return True
            
        except Exception as e:
            self.logger.error(f"模型部署失败: {e}", extra={
                'trading_context': {
                    'error': 'model_deployment_exception',
                    'error_message': str(e)
                }
            })
            return False
    
    