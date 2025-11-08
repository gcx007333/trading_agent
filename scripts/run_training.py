# scripts/run_training.py
import sys
import os
import akshare as ak

# 设置项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.model_trainer import ModelTrainer
from utils.logger import initialize_logging

print("🚀 启动训练脚本...")

# 初始化性能日志记录器
initialize_logging()

def get_all_stock_codes():
    """获取所有沪深股票代码"""
    try:
        # 获取所有A股代码列表
        stock_info_a_code_name = ak.stock_info_a_code_name()
        
        # 过滤掉ST股票和特定板块（可选）
        all_stocks = stock_info_a_code_name['code'].tolist()
        
        print(f"📈 获取到 {len(all_stocks)} 只股票")
        return all_stocks
        
    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}")
        # 返回空列表或默认股票列表
        return []

def main():
    # 获取所有股票代码
    all_stocks = get_all_stock_codes()

    # 如果获取失败，使用备用列表
    if not all_stocks:
        print("⚠️ 使用备用股票列表")
        # 热门A股列表
        all_stocks = ["601615", # 明阳智能
                      "002202", # 金风科技
                      "000400", # 许继电气 
                    "603019", # 中科曙光
                    "002371", # 北方华创
                    "600011", # 华能国际
                    "688027", # 国盾量子
                    "300124", # 汇川技术
                    "002747", # 埃斯顿
                    "603259", # 药明康德
                    "300760", # 迈瑞医疗
                    "600827", # 百联股份
                    "600718", # 东软集团
                    "600588", # 用友网络
                    "601377", # 兴业证券
                    "600303", # 曙光股份
                    "600546", # 山煤国际
                    "300593", # 新雷能
                    "603596", # 伯特利
                    "688981", # 中芯国际
                    "600919", # 江苏银行
                    "600900", # 长江电力
                    "000933", # 神火股份
                    "002128", # 电投能源
                    "600887", # 伊利股份
                    "600600", # 青岛啤酒
                    "600406", # 国电南瑞
                    "600919"  # 江苏银行
                    ]
    
    # 创建训练器
    trainer = ModelTrainer()
    
    # 批量训练
    results = trainer.train_multiple_stocks(all_stocks)
    
    print("训练完成!")

if __name__ == "__main__":
    main()