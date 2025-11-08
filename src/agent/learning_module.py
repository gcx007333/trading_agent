# src/agent/learning_module.py
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime, timedelta
from core.base_agent import BaseAgent
from models.model_trainer import ModelTrainer
from agent.memory_system import MemorySystem

class LearningModule(BaseAgent):
    """学习模块，负责从经验中学习并优化模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("LearningModule", config)
        self.memory_system = None
        self.model_trainer = None
        self.learning_interval = config.get("learning_interval", 3600)  # 学习间隔(秒)
        self.last_learning_time = None
        self.performance_metrics = {}
        
    def initialize(self, symbols:list=None) -> bool:
        """初始化学习模块"""
        try:
            self.memory_system = MemorySystem(self.config.get("memory", {}))
            self.memory_system.initialize()
            
            self.model_trainer = ModelTrainer("config")
            # self.model_trainer.initialize()
            
            self.last_learning_time = datetime.now()
            self.is_running = True
            self.logger.info("学习模块初始化完成")
            return True
        except Exception as e:
            self.logger.error(f"学习模块初始化失败: {e}")
            return False
            
    def learn_from_experience(self, symbol: str = None, force: bool = False) -> bool:
        """从经验中学习"""
        try:
            current_time = datetime.now()
            
            # 检查是否到达学习间隔
            if not force and (current_time - self.last_learning_time).total_seconds() < self.learning_interval:
                return False
                
            self.logger.info("开始从经验中学习...")
            
            # 1. 收集学习数据
            training_data = self._collect_training_data(symbol)
            if not training_data:
                self.logger.warning("没有足够的数据进行学习")
                return False
                
            # 2. 评估当前模型性能
            current_performance = self._evaluate_current_performance(training_data)
            
            # 3. 训练新模型
            new_model_performance = self._train_new_model(training_data)
            
            # 4. 模型比较和选择
            if self._should_update_model(current_performance, new_model_performance):
                self._update_production_model()
                self.logger.info("模型更新完成")
            else:
                self.logger.info("新模型性能未提升，保持当前模型")
                
            self.last_learning_time = current_time
            return True
            
        except Exception as e:
            self.logger.error(f"学习过程失败: {e}")
            return False
            
    def _collect_training_data(self, symbol: str = None) -> List[Dict[str, Any]]:
        """收集训练数据"""
        training_data = []
        
        # 从记忆系统中检索交易经验
        if symbol:
            symbols = [symbol]
        else:
            # 获取所有有记忆的股票
            symbols = self._get_symbols_with_memories()
            
        for sym in symbols[:10]:  # 限制数量避免内存溢出
            memories = self.memory_system.retrieve_memories(
                sym, "trading_experience", limit=100, min_importance=0.3
            )
            
            for memory in memories:
                training_data.append({
                    'symbol': sym,
                    'features': memory['content'].get('features', {}),
                    'outcome': memory['content'].get('outcome', {}),
                    'timestamp': memory['created_at']
                })
                
        return training_data
        
    def _get_symbols_with_memories(self) -> List[str]:
        """获取有记忆的股票列表"""
        # 这里实现从记忆系统获取所有有记忆的股票
        # 简化实现，返回空列表
        return []
        
    def _evaluate_current_performance(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估当前模型性能"""
        # 实现模型性能评估逻辑
        return {
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.7,
            "f1_score": 0.725
        }
        
    def _train_new_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """训练新模型"""
        try:
            # 准备训练数据
            X, y = self._prepare_training_data(training_data)
            
            if len(X) == 0:
                return {}
                
            # 训练模型
            model_performance = self.model_trainer.train_online(X, y)
            return model_performance
            
        except Exception as e:
            self.logger.error(f"训练新模型失败: {e}")
            return {}
            
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]):
        """准备训练数据"""
        X = []
        y = []
        
        for data in training_data:
            features = data['features']
            outcome = data['outcome']
            
            # 特征向量化
            feature_vector = self._vectorize_features(features)
            if feature_vector is not None:
                X.append(feature_vector)
                # 标签编码
                label = self._encode_outcome(outcome)
                y.append(label)
                
        return np.array(X), np.array(y)
        
    def _vectorize_features(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """特征向量化"""
        try:
            # 实现特征向量化逻辑
            feature_vector = []
            
            # 添加技术指标特征
            technical_features = features.get('technical', {})
            for key in ['rsi', 'macd', 'bollinger_upper', 'bollinger_lower']:
                feature_vector.append(technical_features.get(key, 0))
                
            # 添加基本面特征
            fundamental_features = features.get('fundamental', {})
            for key in ['pe_ratio', 'pb_ratio', 'dividend_yield']:
                feature_vector.append(fundamental_features.get(key, 0))
                
            return np.array(feature_vector) if feature_vector else None
            
        except Exception as e:
            self.logger.error(f"特征向量化失败: {e}")
            return None
            
    def _encode_outcome(self, outcome: Dict[str, Any]) -> int:
        """结果编码"""
        # 根据交易结果编码为分类标签
        profit = outcome.get('profit', 0)
        if profit > 0.02:  # 盈利2%以上
            return 2  # 强烈买入
        elif profit > 0:
            return 1  # 买入
        elif profit < -0.02:  # 亏损2%以上
            return -2  # 强烈卖出
        elif profit < 0:
            return -1  # 卖出
        else:
            return 0  # 持有
            
    def _should_update_model(self, current_perf: Dict[str, float], 
                           new_perf: Dict[str, float]) -> bool:
        """判断是否应该更新模型"""
        if not new_perf:
            return False
            
        # 比较F1分数
        current_f1 = current_perf.get('f1_score', 0)
        new_f1 = new_perf.get('f1_score', 0)
        
        # 新模型性能提升超过5%才更新
        return new_f1 > current_f1 * 1.05
        
    def _update_production_model(self):
        """更新生产环境模型"""
        # 实现模型更新逻辑
        self.model_trainer.deploy_model()
        
    def record_experience(self, symbol: str, features: Dict[str, Any], 
                         decision: Dict[str, Any], outcome: Dict[str, Any]):
        """记录交易经验"""
        experience = {
            'features': features,
            'decision': decision,
            'outcome': outcome,
            'timestamp': datetime.now().isoformat()
        }
        
        # 计算经验重要性
        importance = self._calculate_experience_importance(outcome)
        
        # 存储到记忆系统
        self.memory_system.store_memory(
            symbol, "trading_experience", experience, importance
        )
        
    def _calculate_experience_importance(self, outcome: Dict[str, Any]) -> float:
        """计算经验重要性"""
        profit = abs(outcome.get('profit', 0))
        learning_value = outcome.get('learning_value', 0)
        
        # 基于盈亏和学习价值计算重要性
        base_importance = min(profit * 10, 1.0)  # 盈利越多越重要
        learning_boost = learning_value * 0.5
        
        return min(base_importance + learning_boost, 1.0)
        
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行学习操作"""
        operation = data.get("operation")
        
        if operation == "learn":
            success = self.learn_from_experience(
                symbol=data.get("symbol"),
                force=data.get("force", False)
            )
            return {"status": "success" if success else "no_learning"}
        elif operation == "record_experience":
            self.record_experience(**data.get("params", {}))
            return {"status": "success"}
        else:
            return {"status": "error", "message": f"未知操作: {operation}"}
            
    def shutdown(self):
        """关闭学习模块"""
        self.is_running = False
        if self.memory_system:
            self.memory_system.shutdown()
        if self.model_trainer:
            self.model_trainer.shutdown()