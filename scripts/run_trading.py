# scripts/run_trading.py
import sys
import os

# 设置项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# 示例：使用交易层
from src.trading import (
    AccountManager, OrderExecutor, PositionManager, 
    RiskManager, PortfolioManager, BrokerManager
)
from src.utils.logger import initialize_logging
# 初始化性能日志记录器
initialize_logging()

def demo_trading_layer():
    """演示交易层使用"""
    
    # 1. 初始化各个管理器
    account_manager = AccountManager()
    position_manager = PositionManager(account_manager)
    risk_manager = RiskManager(account_manager, position_manager)
    portfolio_manager = PortfolioManager(account_manager, position_manager, risk_manager)
    
    # 2. 初始化券商接口
    broker_manager = BrokerManager()
    broker_config = {
        'broker_type': 'simulation',
        'broker_name': '模拟券商',
        'initial_capital': 100000.0
    }
    broker_manager.add_broker('simulation', broker_config)
    broker_manager.connect_broker('simulation')
    
    # 3. 初始化订单执行器
    order_executor = OrderExecutor(account_manager, broker_manager.get_active_broker())
    
    print("交易层初始化完成")
    
    # 4. 执行一些示例操作
    
    # 查看账户信息
    account_info = account_manager.get_account_info()
    print(f"账户信息: 总资产 {account_info['total_value']:.2f}, 现金 {account_info['current_cash']:.2f}")
    
    # 风险评估
    risk_assessment = risk_manager.assess_portfolio_risk()
    print(f"风险评估: {risk_assessment.overall_risk.value} (分数: {risk_assessment.risk_score:.2f})")
    
    # 下达示例订单
    order_request = {
        'symbol': '000001',
        'action': 'BUY',
        'quantity': 1000,
        'order_type': 'market',
        'reason': '示例买入',
        'strategy': 'demo'
    }
    
    order = order_executor.place_order(order_request)
    print(f"订单状态: {order.status.value}")
    
    # 查看持仓
    positions = position_manager.account_manager.get_positions()
    print(f"当前持仓: {len(positions)} 只股票")
    
    # 投资组合再平衡
    rebalance_actions = portfolio_manager.rebalance_portfolio('default')
    print(f"再平衡操作: {len(rebalance_actions)} 个")

if __name__ == "__main__":
    demo_trading_layer()