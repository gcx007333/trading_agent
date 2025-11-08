# src/agent/investment_report_agent.py

import time
import pandas as pd
import numpy as np
import json
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from models.model_predictor import ModelPredictor
from data.data_collector import DataCollector
from utils.logger import get_logger
from utils.config_loader import ConfigLoader
from openai import OpenAI

class InvestmentReportAgent:
    """
    投资报告生成Agent
    专门用于生成面向抖音的投资分析报告
    """
    
    def __init__(self, config_path: str = "config"):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_all_configs()
        self.logger = get_logger(__name__)
        
        # 初始化核心组件
        self.model_predictor = ModelPredictor(config_path)
        self.data_collector = DataCollector()
        
        # 券商数据源配置
        self.broker_sources = self.config.get('broker_sources', {
            'weekly': [],
            'monthly': [], 
            'long_term': []
        })
        
        # 报告配置
        self.report_config = self.config.get('report').get('report')
        
        # 初始化Azure DeepSeek客户端
        self._init_azure_deepseek_client()
        
        self.logger.info("投资报告Agent初始化完成")

    def _init_azure_deepseek_client(self):
        """初始化Azure DeepSeek客户端"""
        try:
            ai_config = self.config.get('report').get('ai_service')
            
            # 从环境变量获取API密钥
            api_key = os.getenv('DEEPSEEK_AZURE_API_KEY')
            if not api_key:
                self.logger.warning("未找到DEEPSEEK_AZURE_API_KEY环境变量，AI分析功能将不可用")
                self.deepseek_client = None
                return
            
            endpoint = ai_config.get('azure_deepseek').get('endpoint')
            deployment_name = ai_config.get('azure_deepseek').get('deployment_name')
            
            self.deepseek_client = OpenAI(
                base_url=endpoint,
                api_key=api_key
            )
            
            self.deployment_name = deployment_name
            self.logger.info("Azure DeepSeek客户端初始化成功")
            
        except Exception as e:
            self.logger.error(f"初始化Azure DeepSeek客户端失败: {e}")
            self.deepseek_client = None

    def _call_ai_service(self, question: str, max_retries: int = 3) -> str:
        """
        调用Azure DeepSeek-R1模型获取AI分析结果
        """
        if not self.deepseek_client:
            return "AI分析服务未正确配置，请检查DEEPSEEK_AZURE_API_KEY环境变量。"
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"尝试第{attempt+1}次调用DeepSeek API")
                completion = self.deepseek_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    temperature=0.3,  # 较低的温度值，使输出更加确定性
                    max_tokens=4000,    # 控制回复长度
                    timeout=30  # 设置超时
                )
                
                ai_reply = completion.choices[0].message.content
                self.logger.info("成功获取Azure DeepSeek分析结果" )
                return ai_reply.strip()
                
            except Exception as e:
                self.logger.warning(f"第{attempt+1}次调用失败: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 重试等待时间递增
                    self.logger.info(f"{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"DeepSeek API调用失败，已达最大重试次数: {e}")
                    return f"AI分析暂时不可用: {str(e)}"

    def _build_ai_question(self, symbol: str, prediction: Dict) -> str:
        """
        构建咨询AI的问题 - 优化版本
        """
        direction = "上涨" if prediction.get('prediction_type') == 'bullish' else "下跌"
        confidence = prediction.get('confidence', 0)
        current_price = prediction.get('current_price', 0)

        question = f"""
        请分析股票 {symbol} ：
        
        当前情况：
        - 当前价格：{current_price:.2f}元
        - 预测方向：明日{direction}
        - 模型置信度：{confidence:.1%}
        - 上涨概率：{prediction.get('up_probability', 0):.1%}
        - 下跌概率：{prediction.get('down_probability', 0):.1%}
        
        请基于以上信息，从技术面、基本面、市场情绪等角度提供客观分析，并包含适当的风险提示。请直接给出股票分析结论，不要展示推理过程。
        要求：
        1. 分析要简洁明了，适合在投资报告中展示
        2. 避免使用过于专业的术语，用通俗易懂的语言
        3. 必须包含风险提示
        4. 总字数控制在150-200字之间
        5. 保持客观中立，不给出具体的买卖建议
        """
        
        return question

    def get_broker_recommendations(self) -> Dict[str, List[str]]:
        """
        从券商获取推荐股票（模拟实现）
        实际应用中可以从券商研报、财经网站等获取
        """
        # 这里模拟券商推荐，你可以替换为真实的数据源
        recommendations = {
            'weekly': ["601615", # 明阳智能
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
                ],
            'monthly': [ "600827", # 百联股份
                "600718", # 东软集团
                "600588", # 用友网络
                "601377", # 兴业证券
                "600303", # 曙光股份
                ],
            'long_term': ["600546", # 山煤国际
                "300593", # 新雷能
                "603596", # 伯特利
                "688981", # 中芯国际
                "600900", # 长江电力
                "000933", # 神火股份
                "002128", # 电投能源
                "600887", # 伊利股份
                "600600", # 青岛啤酒
                "600406", # 国电南瑞
                "600919"  # 江苏银行
                ]
        }
        
        self.logger.info(f"获取券商推荐: 周推{len(recommendations['weekly'])}只, "
                        f"月推{len(recommendations['monthly'])}只, "
                        f"长期{len(recommendations['long_term'])}只")
        
        return recommendations

    def generate_daily_prediction_report(self) -> Dict[str, Any]:
        """
        生成每日预测报告
        """
        self.logger.info("开始生成每日预测报告")
        
        try:
            # 1. 获取券商推荐
            recommendations = self.get_broker_recommendations()
            
            # 2. 加载预测模型
            all_symbols = list(set(
                recommendations['weekly'] + 
                recommendations['monthly'] + 
                recommendations['long_term']
            ))
            
            if not self.model_predictor.load_models(all_symbols):
                self.logger.error("模型加载失败，无法生成报告")
                return {}
            
            # 3. 批量预测
            prediction_results = {}
            for period, symbols in recommendations.items():
                prediction_results[period] = {}
                for symbol in symbols[:self.report_config['max_stocks_per_category']]:
                    prediction = self.model_predictor.predict_single_stock(symbol)
                    if prediction and prediction['confidence'] >= self.report_config['min_confidence']:
                        prediction_results[period][symbol] = prediction
            
            # 4. 生成报告
            report = self._format_prediction_report(prediction_results, recommendations)
            
            # 5. 保存报告
            self._save_report(report, "daily_prediction")
            
            self.logger.info(f"每日预测报告生成完成，共分析{len(all_symbols)}只股票")
            return report
            
        except Exception as e:
            self.logger.error(f"生成每日预测报告失败: {e}")
            return {}

    def _format_prediction_report(self, predictions: Dict, recommendations: Dict) -> Dict[str, Any]:
        """
        格式化预测报告
        """
        report = {
            'report_id': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generate_time': datetime.now().isoformat(),
            'summary': {},
            'details': {},
            'broker_links': {},
            'ai_analysis': {}
        }
        
        # 生成摘要统计
        for period, period_predictions in predictions.items():
            bullish_count = sum(1 for p in period_predictions.values() 
                              if p.get('prediction_type') == 'bullish')
            total_count = len(period_predictions)
            
            report['summary'][period] = {
                'total_stocks': total_count,
                'bullish_count': bullish_count,
                'bullish_ratio': bullish_count / total_count if total_count > 0 else 0,
                'average_confidence': np.mean([p.get('confidence', 0) for p in period_predictions.values()]) 
                                    if period_predictions else 0
            }
        
        # 生成详细预测
        report['details'] = predictions
        
        # 生成券商链接
        if self.report_config['include_broker_links']:
            report['broker_links'] = self._generate_broker_links(recommendations)
        
        # 生成AI分析
        if self.report_config['include_ai_analysis']:
            report['ai_analysis'] = self._generate_ai_analysis(predictions)
        
        return report

    def _generate_broker_links(self, recommendations: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        生成券商页面链接 - 修正格式
        """
        links = {}
        
        for period, symbols in recommendations.items():
            links[period] = {}
            for symbol in symbols:
                # 根据股票代码判断市场，生成正确的链接格式
                if symbol.startswith('6'):
                    # 上海证券交易所
                    market_prefix_sh = 'sh'
                    market_prefix_sina = 'sh'
                else:
                    # 深圳证券交易所或其他
                    market_prefix_sh = 'sz' 
                    market_prefix_sina = 'sz'
                
                links[period][symbol] = {
                    'eastmoney': f"https://quote.eastmoney.com/{market_prefix_sh}{symbol}.html",
                    'sina': f"https://finance.sina.com.cn/realstock/company/{market_prefix_sina}{symbol}/nc.shtml",
                    'tencent': f"https://gu.qq.com/{market_prefix_sh}{symbol}"
                }
        
        return links

    def _generate_ai_analysis(self, predictions: Dict) -> Dict[str, Any]:
        """
        生成AI分析内容（调用DeepSeek或其他AI）
        """
        ai_analysis = {}
        
        try:
            for period, period_predictions in predictions.items():
                ai_analysis[period] = {}
                
                for symbol, prediction in period_predictions.items():
                    # 构建咨询问题
                    question = self._build_ai_question(symbol, prediction)
                    
                    # 调用AI接口（这里需要你配置实际的AI服务）
                    ai_response = self._call_ai_service(question)
                    
                    ai_analysis[period][symbol] = {
                        'question': question,
                        'answer': ai_response
                    }
        
        except Exception as e:
            self.logger.error(f"生成AI分析失败: {e}")
        
        return ai_analysis

    def _save_report(self, report: Dict, report_type: str):
        """
        保存报告到文件
        """
        try:
            report_dir = "results/investment_reports"
            os.makedirs(report_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_dir}/{report_type}_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"报告已保存: {filename}")
            
            # 同时生成简化版本用于抖音展示
            self._generate_simple_version(report, filename)
            
        except Exception as e:
            self.logger.error(f"保存报告失败: {e}")

    def _generate_simple_version(self, report: Dict, original_filename: str):
        """
        生成简化版本报告，适合抖音展示
        """
        try:
            simple_report = {
                'generate_time': report['generate_time'],
                'summary': report['summary'],
                'top_picks': {}
            }
            
            # 选取每个周期中置信度最高的3只股票
            for period, details in report['details'].items():
                sorted_stocks = sorted(
                    details.items(), 
                    key=lambda x: x[1].get('confidence', 0), 
                    reverse=True
                )[:3]
                
                simple_report['top_picks'][period] = {
                    symbol: {
                        'prediction': data.get('prediction', ''),
                        'confidence': data.get('confidence', 0),
                        'current_price': data.get('current_price', 0)
                    }
                    for symbol, data in sorted_stocks
                }
            
            simple_filename = original_filename.replace('.json', '_simple.json')
            with open(simple_filename, 'w', encoding='utf-8') as f:
                json.dump(simple_report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"简化版报告已保存: {simple_filename}")
            
        except Exception as e:
            self.logger.error(f"生成简化版报告失败: {e}")

    def generate_visual_report(self, report_data: Dict) -> str:
        """
        生成可视化报告（图表和详细表格）
        返回图表文件路径
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 各周期看涨比例
            periods = list(report_data['summary'].keys())
            bullish_ratios = [report_data['summary'][p]['bullish_ratio'] for p in periods]
            
            ax1.bar(periods, bullish_ratios, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            ax1.set_title('各周期推荐股票看涨比例')
            ax1.set_ylabel('看涨比例')
            
            # 2. 置信度分布
            confidences = []
            for period in report_data['details'].values():
                for stock in period.values():
                    confidences.append(stock.get('confidence', 0))
            
            ax2.hist(confidences, bins=10, alpha=0.7, color='#96ceb4')
            ax2.set_title('预测置信度分布')
            ax2.set_xlabel('置信度')
            ax2.set_ylabel('股票数量')
            
            # 3. 各周期股票数量
            stock_counts = [report_data['summary'][p]['total_stocks'] for p in periods]
            ax3.pie(stock_counts, labels=periods, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
            ax3.set_title('各周期推荐股票数量分布')
            
            # 4. 最高置信度股票
            top_stocks = []
            top_confidences = []
            for period, stocks in report_data['details'].items():
                if stocks:
                    top_stock = max(stocks.items(), key=lambda x: x[1].get('confidence', 0))
                    top_stocks.append(f"{top_stock[0]}\n({period})")
                    top_confidences.append(top_stock[1].get('confidence', 0))
            
            ax4.barh(top_stocks, top_confidences, color='#ffcc5c')
            ax4.set_title('各周期最高置信度股票')
            ax4.set_xlabel('置信度')
            
            plt.tight_layout()
            
            # 保存图表
            chart_dir = "results/charts"
            os.makedirs(chart_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = f"{chart_dir}/report_chart_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"可视化图表已保存: {chart_path}")
            
            # 生成详细股票预测表格
            table_path = self._generate_detailed_table(report_data, timestamp)
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"生成可视化报告失败: {e}")
            return ""

    def _generate_detailed_table(self, report_data: Dict, timestamp: str) -> str:
        """
        生成详细的股票预测表格（CSV格式）- 使用自动获取的股票名称
        """
        try:
            table_data = []
            
            # 遍历所有周期的股票预测
            for period, stocks in report_data['details'].items():
                for symbol, prediction in stocks.items():
                    
                    # 获取券商链接
                    broker_links = report_data.get('broker_links', {}).get(period, {}).get(symbol, {})
                    
                    # 获取AI分析结果
                    ai_analysis = report_data.get('ai_analysis', {}).get(period, {}).get(symbol, {})
                    ai_answer = ai_analysis.get('answer', '暂无AI分析') if ai_analysis else '暂无AI分析'
                    
                    # 构建行数据
                    row = {
                        '推荐周期': period,
                        '股票代码': symbol,
                        '股票名称': prediction.get('name', ''),
                        '预测方向': prediction.get('prediction', ''),
                        '预测类型': prediction.get('prediction_type', ''),
                        '上涨概率': f"{prediction.get('up_probability', 0):.2%}",
                        '下跌概率': f"{prediction.get('down_probability', 0):.2%}",
                        '置信度': f"{prediction.get('confidence', 0):.2%}",
                        '当前价格': f"{prediction.get('current_price', 0):.2f}",
                        '开盘价格': f"{prediction.get('current_open', 0):.2f}",
                        '模型准确率': f"{prediction.get('model_accuracy', 0):.2%}",
                        '预测描述': prediction.get('prediction_description', ''),
                        '东方财富链接': broker_links.get('eastmoney', ''),
                        '新浪财经链接': broker_links.get('sina', ''),
                        '腾讯财经链接': broker_links.get('tencent', ''),
                        'DeepSeek分析': ai_answer.replace('\n', ' ')  # 替换换行符以便在表格中显示
                    }
                    
                    table_data.append(row)
            
            # 创建DataFrame
            df = pd.DataFrame(table_data)
            
            # 保存为CSV文件
            table_dir = "results/tables"
            os.makedirs(table_dir, exist_ok=True)
            table_path = f"{table_dir}/stock_predictions_{timestamp}.csv"
            df.to_csv(table_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"详细股票预测表格已保存: {table_path}")
            
            # 同时保存为Excel格式（可选）
            excel_path = f"{table_dir}/stock_predictions_{timestamp}.xlsx"
            df.to_excel(excel_path, index=False)
            self.logger.info(f"Excel格式表格已保存: {excel_path}")
            
            return table_path
            
        except Exception as e:
            self.logger.error(f"生成详细表格失败: {e}")
            return ""

    def _generate_tiktok_summary(self, report: Dict, chart_path: str) -> Dict[str, Any]:
        """
        生成抖音内容摘要 - 更新版本，包含表格路径
        """
        summary = {
            'title': f"每日投资报告 {datetime.now().strftime('%m月%d日')}",
            'key_points': [],
            'hashtags': ['#投资报告', '#股票分析', '#A股', '#财经知识'],
            'chart_path': chart_path,
            'table_path': None
        }
        
        # 生成关键点
        for period, period_summary in report['summary'].items():
            point = {
                'period': period,
                'bullish_ratio': f"{period_summary['bullish_ratio']:.1%}",
                'average_confidence': f"{period_summary['average_confidence']:.1%}",
                'top_stocks': []
            }
            
            # 获取该周期置信度最高的股票
            period_stocks = report['details'].get(period, {})
            if period_stocks:
                top_3 = sorted(period_stocks.items(), 
                            key=lambda x: x[1].get('confidence', 0), 
                            reverse=True)[:3]
                
                for symbol, data in top_3:
                    point['top_stocks'].append({
                        'symbol': symbol,
                        'prediction': data.get('prediction', ''),
                        'confidence': f"{data.get('confidence', 0):.1%}",
                        'current_price': f"{data.get('current_price', 0):.2f}"
                    })
            
            summary['key_points'].append(point)
        
        return summary

    def run_daily_workflow(self):
        """
        运行每日工作流
        """
        self.logger.info("开始执行每日投资报告工作流")
        
        # 1. 生成预测报告
        report = self.generate_daily_prediction_report()
        
        if not report:
            self.logger.error("报告生成失败，工作流终止")
            return
        
        # 2. 生成可视化图表和详细表格
        chart_path = self.generate_visual_report(report)
        
        # 3. 生成抖音内容摘要
        tiktok_summary = self._generate_tiktok_summary(report, chart_path)
        
        self.logger.info("每日投资报告工作流执行完成")
        
        return {
            'report': report,
            'chart_path': chart_path,
            'table_path': tiktok_summary.get('table_path'),
            'tiktok_summary': tiktok_summary
        }

    def _generate_tiktok_summary(self, report: Dict, chart_path: str) -> Dict[str, Any]:
        """
        生成抖音内容摘要
        """
        summary = {
            'title': f"每日投资报告 {datetime.now().strftime('%m月%d日')}",
            'key_points': [],
            'hashtags': ['#投资报告', '#股票分析', '#A股', '#财经知识'],
            'chart_path': chart_path
        }
        
        # 生成关键点
        for period, period_summary in report['summary'].items():
            point = {
                'period': period,
                'bullish_ratio': f"{period_summary['bullish_ratio']:.1%}",
                'average_confidence': f"{period_summary['average_confidence']:.1%}",
                'top_stocks': []
            }
            
            # 获取该周期置信度最高的股票
            period_stocks = report['details'].get(period, {})
            if period_stocks:
                top_3 = sorted(period_stocks.items(), 
                             key=lambda x: x[1].get('confidence', 0), 
                             reverse=True)[:3]
                
                for symbol, data in top_3:
                    point['top_stocks'].append({
                        'symbol': symbol,
                        'prediction': data.get('prediction', ''),
                        'confidence': f"{data.get('confidence', 0):.1%}"
                    })
            
            summary['key_points'].append(point)
        
        return summary