# scripts/run_prediction.py  
# 设置项目根目录
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.model_predictor import ModelPredictor

from src.utils.logger import initialize_logging
# 初始化性能日志记录器
initialize_logging()

def main():
    # 要预测的股票
    symbols = ["601615", # 明阳智能
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
    
    # 创建预测器
    predictor = ModelPredictor()
    
    # 批量预测
    predictions = predictor.predict_multiple_stocks(symbols)
    
    print("预测完成!")
    for symbol, prediction in predictions['details'].items():
        print(f"{symbol}: {prediction['prediction']} (置信度: {prediction['confidence']:.3f}) {prediction['name']}")

if __name__ == "__main__":
    main()