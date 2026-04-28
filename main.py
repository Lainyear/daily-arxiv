"""
Daily arXiv Agent - 主程序入口 / Main entry

每日追踪 arXiv 最新论文，使用 LLM 进行总结和分析 /
Track latest arXiv papers daily and summarize/analyze them with LLMs
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径 / Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, load_env, setup_logging, get_date_string, pick_text


def main():
    """主函数 / Main function"""
    # 加载配置 / Load configuration
    load_env()
    config = load_config()
    logger = setup_logging(config)
    text = lambda zh, en: pick_text(config, zh, en)
    
    logger.info("=" * 60)
    logger.info(text("Daily arXiv Agent 启动", "Daily arXiv Agent started"))
    logger.info(f"{text('日期', 'Date')}: {get_date_string()}")
    logger.info("=" * 60)
    
    try:
        # 第二步 - 实现论文爬取 ✅ / Step 2 - Fetch papers
        logger.info(text("步骤 1: 爬取 arXiv 论文...", "Step 1: Fetching arXiv papers..."))
        from src.crawler.arxiv_fetcher import ArxivFetcher
        fetcher = ArxivFetcher(config)
        
        # 尝试获取论文，如果没找到，逐步放宽条件 / Retry with broader window if no papers are found
        papers = fetcher.fetch_papers(days_back=2)
        
        if not papers:
            logger.warning(text(
                "⚠️  过去2天没有找到符合条件的论文，尝试扩大到7天...",
                "⚠️  No matching papers found in last 2 days, retrying with a 7-day window..."
            ))
            papers = fetcher.fetch_papers(days_back=7)
        
        if papers:
            fetcher.print_paper_summary(papers)
        else:
            logger.warning(text("⚠️  没有找到符合条件的论文", "⚠️  No matching papers found"))
            logger.info(text("💡 提示: 可以尝试以下方法：", "💡 Tips:"))
            logger.info(text("   1. 在 config.yaml 中增加 days_back 或 max_results", "   1. Increase days_back or max_results in config.yaml"))
            logger.info(text("   2. 减少或删除关键词过滤（设置 keywords: []）", "   2. Reduce or remove keyword filters (set keywords: [])"))
            logger.info(text("   3. 修改类别范围", "   3. Broaden category scope"))
            return
        
        # 第三步 - 实现论文总结 ✅ / Step 3 - Summarize papers
        logger.info(text("\n步骤 2: 总结论文...", "\nStep 2: Summarizing papers..."))
        from src.summarizer.paper_summarizer import PaperSummarizer
        
        # 初始化变量，防止LLM失败导致程序崩溃
        summarized_papers = papers
        summarizer = None
        try:
            summarizer = PaperSummarizer(config)
            summarized_papers = summarizer.summarize_papers(papers)
            logger.info(text("✅ 论文总结完成", "✅ Paper summarization completed"))
        except Exception as e:
            logger.error(text(f"论文总结失败: {str(e)}", f"Paper summarization failed: {str(e)}"))
            logger.info(text("继续执行，将用原始论文生成报告...", "Continuing with raw paper info..."))
        
        # --------------------------
        # 🔴 把生成报告的代码移到try块外面，强制执行
        # --------------------------
        logger.info(text("\n生成每日报告...", "\nGenerating daily report..."))
        # 即使LLM失败，也手动生成一个基础报告
        if summarizer:
            report = summarizer.generate_daily_report(summarized_papers)
        else:
            # 手动生成简易报告（防止LLM失败时没文件）
            report = f"# {get_date_string()} arXiv 论文日报\n\n## 今日论文（LLM总结失败，显示原始信息）\n"
            for paper in summarized_papers:
                report += f"### {paper['title']}\n"
                report += f"- 作者: {', '.join(paper['authors'])}\n"
                report += f"- 分类: {paper['categories']}\n"
                report += f"- 链接: {paper['pdf_url']}\n"
                report += f"- 摘要: {paper['summary'][:300]}...\n\n"
        
        # 保存报告（强制创建目录）
        report_path = f"data/summaries/report_{get_date_string()}.md"
        from pathlib import Path
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)  # 强制创建data/summaries
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(text(f"📄 每日报告已保存到: {report_path}", f"📄 Daily report saved to: {report_path}"))
        
        # --------------------------
        # 🔴 新增：保存原始论文数据到data/papers/
        # --------------------------
        import json
        papers_path = f"data/papers/papers_{get_date_string()}.json"
        Path(papers_path).parent.mkdir(parents=True, exist_ok=True)  # 强制创建data/papers
        with open(papers_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        logger.info(text(f"📄 原始论文数据已保存到: {papers_path}", f"📄 Raw paper data saved to: {papers_path}"))
                
        # 第四步 - 实现趋势分析 ✅ / Step 4 - Analyze trends
        logger.info(text("\n步骤 3: 分析研究趋势...", "\nStep 3: Analyzing research trends..."))
        try:
            from src.analyzer.trend_analyzer import TrendAnalyzer
            from src.summarizer.llm_factory import LLMClientFactory
            
            # 创建 LLM 客户端（用于深度分析）/ Create LLM client for deep analysis
            llm_client = LLMClientFactory.create_client(config)
            
            # 加载论文总结 / Load summaries
            from src.utils import load_json
            summaries_data = load_json('data/summaries/latest.json')
            summaries = summaries_data.get('summaries', []) if summaries_data else []
            
            # 创建趋势分析器 / Create trend analyzer
            analyzer = TrendAnalyzer(config, llm_client)
            analysis = analyzer.analyze(papers, summaries)
            
            if analysis:
                analyzer.print_analysis_summary(analysis)
            
        except Exception as e:
            logger.error(text(f"趋势分析失败: {str(e)}", f"Trend analysis failed: {str(e)}"), exc_info=True)
            logger.info(text("继续执行后续步骤...", "Continuing with following steps..."))
        
        logger.info("=" * 60)
        logger.info(text("✅ 所有任务完成！", "✅ All tasks completed!"))
        logger.info("=" * 60)
        logger.info(text("提示: 运行 'python src/web/app.py' 启动 Web 服务查看结果", "Tip: run 'python src/web/app.py' to start the web service"))

        send_arxiv_report_to_email()
        
    except Exception as e:
        logger.error(text(f"❌ 执行出错: {str(e)}", f"❌ Execution failed: {str(e)}"), exc_info=True)
        sys.exit(1)

def send_arxiv_report_to_email():
    """
    读取生成的日报文件，发送到你的QQ邮箱
    """
    # 1. 从 .env 读取你配置的邮箱变量
    load_dotenv()
    sender = os.getenv("EMAIL_FROM")
    receiver = os.getenv("EMAIL_TO")
    auth_code = os.getenv("EMAIL_AUTH_CODE")

    if not all([sender, receiver, auth_code]):
        print("❌ 邮箱配置不完整，跳过发送邮件")
        return

    # 2. 配置 QQ 邮箱 SMTP 服务器
    smtp_server = "smtp.qq.com"
    smtp_port = 465

    # 3. 创建邮件
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = "📚 每日ArXiv论文中文日报"

    # 邮件正文
    body = "今日ArXiv论文日报已生成，附件包含：\n1. 中文总结报告 (Markdown格式)\n2. 原始论文数据 (JSON格式)"
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # 4. 自动查找并添加今天生成的文件作为附件
    date_str = get_date_string() # 复用你代码里已有的日期函数
    files_to_send = [
        f"data/summaries/report_{date_str}.md",
        f"data/papers/papers_{date_str}.json"
    ]

    for file_path in files_to_send:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            with open(file_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=filename)
            part["Content-Disposition"] = f'attachment; filename="{filename}"'
            msg.attach(part)
            print(f"✅ 已添加附件: {filename}")
        else:
            print(f"⚠️ 未找到文件: {file_path}")

    # 5. 发送邮件
    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender, auth_code)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        print("✅ 邮件发送成功！")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
if __name__ == "__main__":
    main()
