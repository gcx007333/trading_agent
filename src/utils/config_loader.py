# src/utils/config_loader.py
import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """数据库配置"""
    path: str = "data/database/trading.db"
    backup_enabled: bool = True
    backup_interval: int = 3600  # 秒

@dataclass
class DataSourceConfig:
    """数据源配置"""
    akshare_enabled: bool = True
    tushare_enabled: bool = False
    tushare_token: Optional[str] = None
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 秒

@dataclass
class ModelConfig:
    """模型配置"""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 10

@dataclass
class TradingConfig:
    """交易配置"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    transfer_fee_rate: float = 0.00002
    max_position_size: float = 0.2  # 单票最大仓位
    max_drawdown: float = 0.1  # 最大回撤限制

@dataclass
class RiskConfig:
    """风险配置"""
    stop_loss: float = 0.05  # 止损比例，当股价下跌超过5%时触发止损
    take_profit: float = 0.15  # 止盈比例，当股价上涨超过15%时触发止盈
    max_daily_loss: float = 0.03  # 单日最大亏损，当日亏损超过总资产3%时触发风控
    position_sizing: str = "kelly"  # 仓位分配策略，可选：kelly(凯利公式)、fixed(固定比例)、volatility(波动率调整)
    
    # 追加的参数
    max_drawdown: float = 0.1  # 最大回撤限制，投资组合从峰值回落的最大允许幅度(10%)
    max_position_size: float = 0.2  # 单票最大仓位，单个股票在投资组合中的最大权重(20%)
    max_sector_exposure: float = 0.3  # 最大行业暴露，单个行业在投资组合中的最大权重(30%)
    concentration_limit: float = 0.6  # 组合集中度限制，前5大持仓占总投资组合的最大比例(60%)
    min_cash_ratio: float = 0.1  # 最小现金比例，投资组合中必须保持的最低现金比例(10%)
    risk_per_trade: float = 0.02  # 每笔交易风险，单次交易允许损失的最大资金比例(2%)
    volatility_threshold: float = 0.25  # 波动率阈值，年化波动率超过25%时触发风险警告
    extreme_risk_score: float = 8.0  # 极高风险分数阈值，风险评分超过8.0为极端风险
    high_risk_score: float = 6.0  # 高风险分数阈值，风险评分超过6.0为高风险
    medium_risk_score: float = 3.0  # 中等风险分数阈值，风险评分超过3.0为中等风险

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "results/logs/trading_system.log"
    max_file_size: int = 100  # MB
    backup_count: int = 5

@dataclass
class SystemConfig:
    """系统配置"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

class ConfigLoader:
    """
    配置加载器
    支持YAML、JSON格式，环境变量覆盖，配置验证
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._create_default_configs()
    
    def _create_default_configs(self):
        """创建默认配置文件"""
        default_configs = {
            "system.yaml": {
                "database": {
                    "path": "data/database/trading.db",
                    "backup_enabled": True,
                    "backup_interval": 3600
                },
                "data_source": {
                    "akshare_enabled": True,
                    "tushare_enabled": False,
                    "tushare_token": None,
                    "cache_enabled": True,
                    "cache_ttl": 3600
                },
                "model": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "early_stopping_rounds": 10
                },
                "trading": {
                    "initial_capital": 100000.0,
                    "commission_rate": 0.0003,
                    "stamp_tax_rate": 0.001,
                    "transfer_fee_rate": 0.00002,
                    "max_position_size": 0.2,
                    "max_drawdown": 0.1
                },
                "risk": {
                    "stop_loss": 0.05,  # 止损比例，股价下跌5%触发止损
                    "take_profit": 0.15,  # 止盈比例，股价上涨15%触发止盈
                    "max_daily_loss": 0.03,  # 单日最大亏损，日亏损不超过总资产3%
                    "position_sizing": "kelly",  # 仓位策略：kelly/凯利公式
                    # 追加的参数
                    "max_drawdown": 0.1,  # 最大回撤限制，组合回撤不超过10%
                    "max_position_size": 0.2,  # 单票最大仓位，单股票权重不超过20%
                    "max_sector_exposure": 0.3,  # 最大行业暴露，单行业权重不超过30%
                    "concentration_limit": 0.6,  # 组合集中度，前5大持仓不超过60%
                    "min_cash_ratio": 0.1,  # 最小现金比例，保持至少10%现金
                    "risk_per_trade": 0.02,  # 每笔交易风险，单次交易风险不超过2%
                    "volatility_threshold": 0.25,  # 波动率阈值，年化波动率25%警告
                    "extreme_risk_score": 8.0,  # 极端风险阈值，评分>8.0为极端风险
                    "high_risk_score": 6.0,  # 高风险阈值，评分>6.0为高风险
                    "medium_risk_score": 3.0  # 中等风险阈值，评分>3.0为中等风险
                },
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "file_path": "results/logs/trading_system.log",
                    "max_file_size": 100,
                    "backup_count": 5
                }
            },
            "feature_config.yaml": {
                "target_type": "open_close",
                "technical_indicators": {
                    "trend": True,
                    "momentum": True,
                    "volatility": True,
                    "volume": True,
                    "cycle": False
                },
                "feature_groups": {
                    "price_features": True,
                    "volume_features": True,
                    "technical_features": True,
                    "statistical_features": True,
                    "time_features": True
                },
                "lag_periods": [1, 2, 3, 5, 10, 20],
                "rolling_windows": [5, 10, 20, 60],
                "create_interactions": True
            },
            "trading_strategies.yaml": {
                "strategies": {
                    "momentum": {
                        "enabled": True,
                        "lookback_period": 20,
                        "holding_period": 5
                    },
                    "mean_reversion": {
                        "enabled": True,
                        "lookback_period": 60,
                        "zscore_threshold": 2.0
                    },
                    "breakout": {
                        "enabled": False,
                        "resistance_period": 20,
                        "confirmation_bars": 2
                    }
                }
            }
        }
        
        for filename, config in default_configs.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                self._save_config(config_path, config)
                logger.info(f"创建默认配置文件: {config_path}")
    
    def _save_config(self, filepath: Path, config: Dict):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            if filepath.suffix == '.yaml':
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(config, f, indent=2, ensure_ascii=False)
    
    @lru_cache(maxsize=1)
    def load_system_config(self) -> SystemConfig:
        """加载系统配置"""
        config_file = self.config_dir / "system.yaml"
        config_dict = self._load_config_file(config_file)
        
        # 应用环境变量覆盖
        config_dict = self._apply_environment_overrides(config_dict)
        
        # 转换为数据类
        return self._dict_to_system_config(config_dict)
    
    def load_feature_config(self) -> Dict[str, Any]:
        """加载特征配置"""
        config_file = self.config_dir / "feature_config.yaml"
        return self._load_config_file(config_file)
    
    def load_strategy_config(self) -> Dict[str, Any]:
        """加载策略配置"""
        config_file = self.config_dir / "trading_strategies.yaml"
        return self._load_config_file(config_file)
    
    def load_agent_config(self) -> Dict[str, Any]:
        """加载agent配置"""
        config_file = self.config_dir / "agent_config.yaml"
        return self._load_config_file(config_file)
    
    def load_report_config(self) -> Dict[str, Any]:
        """加报告配置"""
        config_file = self.config_dir / "report_config.yaml"
        return self._load_config_file(config_file)
    
    def load_all_configs(self) -> Dict[str, Any]:
        """加载所有配置"""
        return {
            "system": self.load_system_config(),
            "features": self.load_feature_config(),
            "strategies": self.load_strategy_config(),
            "agent": self.load_agent_config(),
            "report": self.load_report_config()
        }
    
    def _load_config_file(self, filepath: Path) -> Dict[str, Any]:
        """加载配置文件"""
        if not filepath.exists():
            logger.warning(f"配置文件不存在: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.suffix == '.yaml':
                    return yaml.safe_load(f) or {}
                else:
                    return json.load(f) or {}
        except Exception as e:
            logger.error(f"加载配置文件失败 {filepath}: {e}")
            return {}
    
    def _apply_environment_overrides(self, config_dict: Dict) -> Dict:
        """应用环境变量覆盖"""
        env_mappings = {
            "TRADING_INITIAL_CAPITAL": ["trading", "initial_capital"],
            "TRADING_COMMISSION_RATE": ["trading", "commission_rate"],
            "DATABASE_PATH": ["database", "path"],
            "LOG_LEVEL": ["logging", "level"]
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(config_dict, config_path, env_value)
                logger.debug(f"环境变量覆盖: {env_var} -> {config_path}")
        
        return config_dict
    
    def _set_nested_value(self, config_dict: Dict, path: list, value: Any):
        """设置嵌套字典的值"""
        current = config_dict
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = self._convert_value_type(value, type(current.get(path[-1], str)))
    
    def _convert_value_type(self, value: str, target_type: type) -> Any:
        """转换值类型"""
        try:
            if target_type == bool:
                return value.lower() in ('true', '1', 'yes', 'y')
            elif target_type == int:
                return int(value)
            elif target_type == float:
                return float(value)
            else:
                return value
        except (ValueError, TypeError):
            return value
    
    def _dict_to_system_config(self, config_dict: Dict) -> SystemConfig:
        """字典转换为系统配置对象"""
        return SystemConfig(
            database=DatabaseConfig(**config_dict.get('database', {})),
            data_source=DataSourceConfig(**config_dict.get('data_source', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            trading=TradingConfig(**config_dict.get('trading', {})),
            risk=RiskConfig(**config_dict.get('risk', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def update_config(self, config_type: str, updates: Dict[str, Any]):
        """更新配置"""
        config_files = {
            "system": "system.yaml",
            "features": "feature_config.yaml",
            "strategies": "trading_strategies.yaml",
            "agent": "agent_config.yaml"
        }
        
        if config_type not in config_files:
            raise ValueError(f"不支持的配置类型: {config_type}")
        
        config_file = self.config_dir / config_files[config_type]
        current_config = self._load_config_file(config_file)
        
        # 深度合并配置
        merged_config = self._deep_merge(current_config, updates)
        
        # 保存更新后的配置
        self._save_config(config_file, merged_config)
        logger.info(f"更新配置: {config_type}")
        
        # 清除缓存
        self.load_system_config.cache_clear()
    
    def _deep_merge(self, base: Dict, updates: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        
        for key, value in updates.items():
            if (key in result and isinstance(result[key], dict) 
                and isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

# 全局配置实例
_config_loader = ConfigLoader()

def get_system_config() -> SystemConfig:
    """获取系统配置（单例）"""
    return _config_loader.load_system_config()

def get_feature_config() -> Dict[str, Any]:
    """获取特征配置"""
    return _config_loader.load_feature_config()

def update_system_config(updates: Dict[str, Any]):
    """更新系统配置"""
    _config_loader.update_config("system", updates)