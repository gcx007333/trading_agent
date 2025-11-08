# src/models/model_predictor.py
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from typing import Dict, Tuple, Optional, Any

from data.data_collector import DataCollector
from data.feature_engineer import FeatureEngineer
from utils.config_loader import ConfigLoader
from utils.logger import get_logger, debug_logger, performance_logger

# 使用项目统一的日志工具
logger = get_logger(__name__)

class ModelPredictor:
    """
    A股模型预测器
    负责加载训练好的模型并进行预测
    """
    
    def __init__(self, config_path="config"):
        config_loader = ConfigLoader(config_path)
        self.config = config_loader.load_all_configs()
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer(self.config['features'])
        self.model_registry = {}
        self._is_ready = False
        
        logger.info("模型预测器初始化完成", extra={
            'trading_context': {
                'action': 'predictor_initialized',
                'config_path': config_path
            }
        })

    def initialize(self, symbols: list = None) -> bool:
        """初始化模型预测器"""
        try:
            # 尝试加载模型
            self._is_ready = self.load_models(symbols)
            return self._is_ready
        except Exception as e:
            logger.error(f"模型预测器初始化失败: {e}")
            return False

    def load_models(self, symbols: list) -> bool:
        """
        批量加载多个模型
        根据传入的股票代码列表加载对应的模型
        
        Args:
            symbols: 股票代码列表，如 ['000001', '000002', '600036']
        
        Returns:
            bool: 是否成功加载至少一个模型
        """
        logger.info("开始批量加载模型", extra={
            'trading_context': {
                'action': 'batch_model_loading_start',
                'symbol_count': len(symbols)
            }
        })
        
        try:
            if not symbols:
                logger.error("未提供股票代码列表，无法加载模型", extra={
                    'trading_context': {
                        'error': 'no_symbols_provided'
                    }
                })
                return False
            
            logger.info("准备加载模型列表", extra={
                'trading_context': {
                    'action': 'models_identified',
                    'model_count': len(symbols),
                    'symbols': symbols
                }
            })
            
            # 批量加载模型
            success_count = 0
            failed_models = []
            
            for symbol in symbols:
                if self.load_model(symbol):
                    success_count += 1
                else:
                    failed_models.append(symbol)
            
            # 记录加载结果
            logger.info("批量加载模型完成", extra={
                'trading_context': {
                    'action': 'batch_model_loading_complete',
                    'total_models': len(symbols),
                    'success_count': success_count,
                    'failed_count': len(failed_models),
                    'success_rate': success_count / len(symbols) if symbols else 0
                }
            })
            
            if failed_models:
                logger.warning("部分模型加载失败", extra={
                    'trading_context': {
                        'warning': 'partial_model_loading_failure',
                        'failed_models_count': len(failed_models),
                        'failed_models': failed_models
                    }
                })
            
            # 如果至少成功加载了一个模型，就认为是就绪状态
            is_ready = success_count > 0
            
            if is_ready:
                logger.info("模型预测器就绪", extra={
                    'trading_context': {
                        'action': 'predictor_ready',
                        'loaded_models_count': success_count,
                        'ready_status': is_ready
                    }
                })
            else:
                logger.error("模型预测器未就绪：没有成功加载任何模型", extra={
                    'trading_context': {
                        'error': 'predictor_not_ready',
                        'reason': 'no_models_loaded_successfully'
                    }
                })
            
            return is_ready
            
        except Exception as e:
            logger.error("批量加载模型失败", extra={
                'trading_context': {
                    'error': 'batch_model_loading_failed',
                    'error_message': str(e)
                }
            })
            return False
        
    def load_model(self, symbol: str) -> bool:
        """
        加载指定股票的模型
        """
        logger.info("开始加载模型", extra={
            'trading_context': {
                'symbol': symbol,
                'action': 'model_loading_start'
            }
        })
        
        try:
            model_path = f"models/xgboost/{symbol}/model.json"
            metadata_path = f"models/xgboost/{symbol}/metadata.json"
            
            if not os.path.exists(model_path):
                logger.error("模型文件不存在", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'error': 'model_file_not_found',
                        'model_path': model_path
                    }
                })
                return False
            
            # 加载模型
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.model_registry[symbol] = {
                'model': model,
                'metadata': metadata,
                'feature_columns': metadata.get('feature_columns', [])
            }
            
            accuracy = metadata.get('accuracy', '未知')
            feature_count = len(metadata.get('feature_columns', []))
            
            logger.info("成功加载模型", extra={
                'trading_context': {
                    'symbol': symbol,
                    'action': 'model_loaded_success',
                    'accuracy': accuracy,
                    'feature_count': feature_count,
                    'training_date': metadata.get('training_date', '未知')
                }
            })
            
            return True
            
        except Exception as e:
            logger.error("加载模型失败", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'model_loading_failed',
                    'error_message': str(e)
                }
            })
            return False
        
    def is_ready(self) -> bool:
            """检查模型是否就绪"""
            return self._is_ready
    """
    A股模型预测器 - 添加predict方法以兼容DecisionMaker
    """
    
    # 现有的初始化代码保持不变...
    
    def predict(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        predict方法 - 供DecisionMaker调用
        接口兼容Agent架构
        """
        """使用模型进行预测"""
        if not self._is_ready:
            raise Exception("模型未就绪，无法进行预测")
        
        try:
            # 使用现有的predict_single_stock方法
            prediction_result = self.predict_single_stock(symbol)
            
            if prediction_result is None:
                return self._get_default_prediction(symbol)
            
            # 转换为DecisionMaker需要的格式
            return self._format_prediction_for_decision_maker(prediction_result, market_data)
            
        except Exception as e:
            logger.error(f"预测失败 {symbol}: {e}")
            return self._get_default_prediction(symbol)
    
    def _format_prediction_for_decision_maker(self, prediction_result: Dict[str, Any], 
                                            market_data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化预测结果供DecisionMaker使用"""
        # 确定方向
        if prediction_result['prediction_type'] == "bullish":
            direction = "buy"
        else:
            direction = "sell"
        
        # 获取当前价格
        current_price = market_data.get('current_price', 0)
        if current_price == 0:
            current_price = prediction_result.get('current_price', 0)
        
        # 计算预期价格（基于置信度和方向）
        confidence = prediction_result['confidence']
        if direction == "buy":
            expected_change = 0.02 * confidence  # 基于置信度的预期涨幅
        else:
            expected_change = -0.02 * confidence  # 基于置信度的预期跌幅
        
        expected_price = current_price * (1 + expected_change)
        
        return {
            "symbol": prediction_result['symbol'],
            "direction": direction,
            "confidence": confidence,
            "prediction_proba": [
                prediction_result['down_probability'],
                prediction_result['up_probability']
            ],
            "expected_price": expected_price,
            "expected_change": expected_change,
            "timestamp": prediction_result['prediction_time'],
            "model_accuracy": prediction_result['model_accuracy'],
            "raw_prediction": prediction_result  # 保留原始预测信息
        }
    
    def _get_default_prediction(self, symbol: str) -> Dict[str, Any]:
        """获取默认预测结果"""
        return {
            "symbol": symbol,
            "direction": "hold",
            "confidence": 0.0,
            "prediction_proba": [0.5, 0.5],
            "expected_price": 0,
            "expected_change": 0,
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_accuracy": 0.5,
            "reason": "预测失败"
        }
    
    # 现有的其他方法保持不变...
    
    def predict_single_stock(self, symbol: str) -> Optional[Dict]:
        """
        预测单只股票 - 增加数据获取长度
        """
        logger.info("开始预测股票", extra={
            'trading_context': {
                'symbol': symbol,
                'action': 'prediction_start'
            }
        })
        
        start_time = pd.Timestamp.now()
        
        try:
            # 1. 检查模型是否已加载
            if symbol not in self.model_registry:
                if not self.load_model(symbol):
                    return None
            
            model_info = self.model_registry[symbol]
            # 2. 获取更长的历史数据以支持季度特征计算
            # quarter_lag_20 可能需要 20*60 ≈ 1200 个交易日的数据
            debug_logger.log_data_processing(symbol, "data_fetch", {
                "days": 1200,  # 增加到1200天
                "action": "prediction_data_extended"
            })
            
            recent_data = self.data_collector.download_recent_data(symbol, days=1200)
            if recent_data is None:
                logger.error("获取预测数据失败", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'error': 'prediction_data_fetch_failed'
                    }
                })
                return None
            
            # 3. 特征工程
            debug_logger.log_data_processing(symbol, "feature_engineering", {
                "data_points": len(recent_data),
                "action": "prediction_features"
            })
            
            featured_data = self.feature_engineer.create_features(
                recent_data, target_type="close_close", for_prediction=True
            )
            
            # 其余代码保持不变...
            
            if featured_data is None:
                logger.error("特征工程失败", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'error': 'feature_engineering_failed'
                    }
                })
                return None
            
            # 4. 准备预测特征
            prediction_features = self._prepare_prediction_features(
                featured_data, model_info['feature_columns'], symbol
            )
            
            if prediction_features is None:
                return None
            
            # 5. 进行预测
            prediction_result = self._make_prediction(
                model_info['model'], prediction_features, symbol
            )

            if prediction_result:
                duration = (pd.Timestamp.now() - start_time).total_seconds()
                
                # 修复：传递字典参数而不是多个位置参数
                performance_logger.log_prediction({
                    'symbol': symbol,
                    'prediction': prediction_result['prediction'],
                    'confidence': prediction_result['confidence'],
                    'up_probability': prediction_result['up_probability'],
                    'duration_seconds': duration,
                    'model_accuracy': prediction_result['model_accuracy']
                })
                
                logger.info("股票预测完成", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'action': 'prediction_complete',
                        'prediction': prediction_result['prediction'],
                        'confidence': prediction_result['confidence'],
                        'duration_seconds': duration
                    }
                })

            return prediction_result
            
        except Exception as e:
            logger.error("预测股票失败", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'prediction_failed',
                    'error_message': str(e),
                    'duration_seconds': (pd.Timestamp.now() - start_time).total_seconds()
                }
            })
            return None
    
    def predict_multiple_stocks(self, symbols: list) -> Dict:
        """
        批量预测多只股票
        """
        logger.info("开始批量预测", extra={
            'trading_context': {
                'action': 'batch_prediction_start',
                'symbol_count': len(symbols),
                'symbols': symbols
            }
        })
        
        start_time = pd.Timestamp.now()
        predictions = {}
        
        for symbol in symbols:
            prediction = self.predict_single_stock(symbol)
            predictions[symbol] = prediction
        
        # 生成预测报告
        report = self._generate_prediction_report(predictions)
        
        duration = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info("批量预测完成", extra={
            'trading_context': {
                'action': 'batch_prediction_complete',
                'total_stocks': len(symbols),
                'successful_predictions': report['successful_predictions'],
                'failed_predictions': report['failed_predictions'],
                'average_confidence': report['average_confidence'],
                'duration_seconds': duration
            }
        })
        
        return report
    
    def _prepare_prediction_features(self, data, feature_columns, symbol):
        """
        准备预测特征 - 增强错误处理和日志
        """
        try:
            # 使用最新的数据点
            latest_data = data.iloc[-1:]
            
            logger.debug("原始数据特征", extra={
                'trading_context': {
                    'symbol': symbol,
                    'available_features_sample': list(data.columns)[:10],
                    'data_shape': data.shape
                }
            })

            # 检查模型元数据，看是否有特征重要性信息
            model_info = self.model_registry[symbol]
            metadata = model_info.get('metadata', {})
            feature_importance = metadata.get('feature_importance', {})
            
            # 检查数据是否包含所需特征
            missing_features = []
            feature_data = {}
            
            for feature in feature_columns:
                if feature in latest_data.columns:
                    feature_value = latest_data[feature].values[0]
                    if pd.isna(feature_value) or np.isinf(feature_value):
                        # 对于重要特征，尝试使用替代值
                        if feature_importance.get(feature, 0) > 0.01:  # 重要性阈值
                            logger.warning("重要特征值无效，使用回退值", extra={
                                'trading_context': {
                                    'symbol': symbol,
                                    'warning': 'important_feature_invalid',
                                    'feature': feature,
                                    'importance': feature_importance.get(feature, 0)
                                }
                            })
                        feature_data[feature] = 0
                    else:
                        feature_data[feature] = feature_value
                else:
                    feature_data[feature] = 0
                    missing_features.append(feature)
                    
                    # 记录特征重要性
                    importance = feature_importance.get(feature, 0)
                    if importance > 0.05:  # 高重要性特征缺失
                        logger.error("高重要性特征缺失", extra={
                            'trading_context': {
                                'symbol': symbol,
                                'error': 'high_importance_feature_missing',
                                'feature': feature,
                                'importance': importance
                            }
                        })
            
            # 一次性创建DataFrame
            matched_features = pd.DataFrame([feature_data], index=latest_data.index)
            
            if missing_features:
                logger.warning("发现缺失特征", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'warning': 'missing_features',
                        'missing_count': len(missing_features),
                        'missing_features': missing_features,
                        'required_features_sample': feature_columns[:5]
                    }
                })
                
                # 记录详细的特征对比
                available_features = set(data.columns)
                required_features = set(feature_columns)
                logger.debug("特征对比详情", extra={
                    'trading_context': {
                        'symbol': symbol,
                        'available_features_count': len(available_features),
                        'required_features_count': len(required_features),
                        'intersection_count': len(available_features & required_features)
                    }
                })
            
            debug_logger.log_data_processing(symbol, "prediction_features", {
                "feature_count": len(feature_columns),
                "missing_features": len(missing_features),
                "data_shape": matched_features.shape,
                "available_data_points": len(data)
            })
            
            return matched_features
            
        except Exception as e:
            logger.error("准备预测特征失败", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'prediction_features_preparation_failed',
                    'error_message': str(e),
                    'feature_columns_sample': feature_columns[:5] if feature_columns else []
                }
            })
            return None
    
    def _make_prediction(self, model, features, symbol):
        """
        进行模型预测 - 更新预测描述
        """
        try:
            debug_logger.log_model_training(symbol, {
                "action": "making_prediction",
                "feature_shape": features.shape
            })
            
            # 预测概率
            prediction_proba = model.predict_proba(features)[0]
            up_probability = prediction_proba[1]
            down_probability = prediction_proba[0]
            
            # 确定预测方向 - 更新描述
            if up_probability > down_probability:
                prediction = "📈 明日上涨"  # 修改描述
                confidence = up_probability
                prediction_type = "bullish"
            else:
                prediction = "📉 明日下跌"  # 修改描述
                confidence = down_probability
                prediction_type = "bearish"
            
            # 获取当前价格信息
            current_data = self.data_collector.get_current_price(symbol)
            
            result = {
                'symbol': symbol,
                'name': current_data.get('name', '') if current_data else '',
                'prediction': prediction,
                'prediction_type': prediction_type,
                'up_probability': float(up_probability),
                'down_probability': float(down_probability),
                'confidence': float(confidence),
                'current_price': current_data.get('current', 0) if current_data else 0,
                'current_open': current_data.get('open', 0) if current_data else 0,
                'prediction_time': pd.Timestamp.now().isoformat(),
                'model_accuracy': self.model_registry[symbol]['metadata'].get('accuracy', 0),
                'prediction_description': '明日收盘相对今日收盘的涨跌'  # 添加描述
            }
            
            # 保存预测结果
            self._save_prediction_result(result)
            
            debug_logger.log_model_training(symbol, {
                "action": "prediction_complete",
                "prediction": prediction,
                "confidence": confidence,
                "up_probability": up_probability,
                "down_probability": down_probability
            })
            
            return result
            
        except Exception as e:
            logger.error("模型预测失败", extra={
                'trading_context': {
                    'symbol': symbol,
                    'error': 'model_prediction_failed',
                    'error_message': str(e)
                }
            })
            return None
    
    def _save_prediction_result(self, prediction_result):
        """
        保存预测结果
        """
        try:
            symbol = prediction_result['symbol']
            predictions_dir = f"results/predictions/{symbol}"
            os.makedirs(predictions_dir, exist_ok=True)
            
            # 按日期保存
            date_str = pd.Timestamp.now().strftime("%Y%m%d")
            file_path = f"{predictions_dir}/prediction_{date_str}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(prediction_result, f, indent=2, ensure_ascii=False)
                
            debug_logger.log_data_processing(symbol, "prediction_saved", {
                "file_path": file_path,
                "prediction": prediction_result['prediction']
            })
            
        except Exception as e:
            logger.error("保存预测结果失败", extra={
                'trading_context': {
                    'error': 'prediction_save_failed',
                    'symbol': prediction_result.get('symbol', 'unknown'),
                    'error_message': str(e)
                }
            })
    
    def _generate_prediction_report(self, predictions):
        """
        生成预测报告
        """
        try:
            successful_predictions = {k: v for k, v in predictions.items() if v is not None}
            failed_predictions = {k: v for k, v in predictions.items() if v is None}
            
            if successful_predictions:
                confidences = [p['confidence'] for p in successful_predictions.values()]
                average_confidence = np.mean(confidences)
                bullish_count = sum(1 for p in successful_predictions.values() 
                                   if p['prediction_type'] == "bullish")
                bearish_count = sum(1 for p in successful_predictions.values() 
                                   if p['prediction_type'] == "bearish")
            else:
                average_confidence = 0
                bullish_count = 0
                bearish_count = 0
            
            report = {
                'total_stocks': len(predictions),
                'successful_predictions': len(successful_predictions),
                'failed_predictions': len(failed_predictions),
                'average_confidence': float(average_confidence),
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'bullish_ratio': bullish_count / len(successful_predictions) if successful_predictions else 0,
                'prediction_date': pd.Timestamp.now().isoformat(),
                'details': successful_predictions,
                'prediction_target': '明日收盘相对今日收盘的涨跌'  # 添加预测目标说明
            }
            
            # 保存报告
            report_dir = "results"
            os.makedirs(report_dir, exist_ok=True)
            report_path = f"{report_dir}/prediction_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # 记录性能指标
            performance_logger.log_performance({
                'batch_prediction_stats': {
                    'total_stocks': report['total_stocks'],
                    'success_rate': report['successful_predictions'] / report['total_stocks'],
                    'average_confidence': report['average_confidence'],
                    'bullish_ratio': report['bullish_ratio']
                }
            })
            
            logger.info("预测报告已保存", extra={
                'trading_context': {
                    'action': 'prediction_report_saved',
                    'report_path': report_path,
                    'successful_predictions': report['successful_predictions'],
                    'failed_predictions': report['failed_predictions'],
                    'average_confidence': report['average_confidence'],
                    'bullish_count': report['bullish_count']
                }
            })
            
            return report
            
        except Exception as e:
            logger.error("生成预测报告失败", extra={
                'trading_context': {
                    'error': 'prediction_report_generation_failed',
                    'error_message': str(e)
                }
            })
            return {}
    
    def get_prediction_summary(self) -> Dict:
        """
        获取预测器状态摘要
        """
        loaded_models = list(self.model_registry.keys())
        
        summary = {
            'loaded_models_count': len(loaded_models),
            'loaded_models': loaded_models,
            'model_details': {}
        }
        
        for symbol, model_info in self.model_registry.items():
            metadata = model_info['metadata']
            summary['model_details'][symbol] = {
                'accuracy': metadata.get('accuracy'),
                'feature_count': len(model_info['feature_columns']),
                'training_date': metadata.get('training_date'),
                'model_type': metadata.get('model_type'),
                'target_description': metadata.get('target', '未知')  # 添加目标描述
            }
        
        logger.debug("获取预测器状态摘要", extra={
            'trading_context': {
                'action': 'predictor_summary',
                'loaded_models_count': len(loaded_models)
            }
        })
        
        return summary
    
    def clear_model_cache(self, symbol: str = None) -> bool:
        """
        清除模型缓存
        """
        try:
            if symbol:
                if symbol in self.model_registry:
                    del self.model_registry[symbol]
                    logger.info("清除单个模型缓存", extra={
                        'trading_context': {
                            'action': 'model_cache_cleared',
                            'symbol': symbol
                        }
                    })
                    return True
                else:
                    logger.warning("模型未在缓存中", extra={
                        'trading_context': {
                            'warning': 'model_not_in_cache',
                            'symbol': symbol
                        }
                    })
                    return False
            else:
                # 清除所有缓存
                cleared_count = len(self.model_registry)
                self.model_registry.clear()
                logger.info("清除所有模型缓存", extra={
                    'trading_context': {
                        'action': 'all_models_cache_cleared',
                        'cleared_count': cleared_count
                    }
                })
                return True
                
        except Exception as e:
            logger.error("清除模型缓存失败", extra={
                'trading_context': {
                    'error': 'model_cache_clear_failed',
                    'symbol': symbol or 'all',
                    'error_message': str(e)
                }
            })
            return False