# scripts/run_daily_report.py

#!/usr/bin/env python3
"""
每日投资报告生成脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.investment_report_agent import InvestmentReportAgent
from src.utils.logger import initialize_logging

def main():
    print("🚀 开始生成每日投资报告...")
    
    try:
        # 设置日志
        initialize_logging()
        agent = InvestmentReportAgent()
        result = agent.run_daily_workflow()
        
        if result:
            print("✅ 投资报告生成成功！")
            print(f"📈 生成时间: {result['report']['generate_time']}")
            print(f"🎨 图表文件: {result['chart_path']}")
            
            # 打印摘要信息
            summary = result['tiktok_summary']
            print(f"📱 视频标题: {summary['title']}")
            print("🔑 关键数据:")
            for point in summary['key_points']:
                print(f"   {point['period']}: 看涨比例 {point['bullish_ratio']}, "
                      f"平均置信度 {point['average_confidence']}")
            
        else:
            print("❌ 投资报告生成失败")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 脚本执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()