import datetime

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown

from agentic_stocks_trading import logger
from agentic_stocks_trading.application.agent_service.nodes_and_runners import run_analyst, setup_trading_analysts
from agentic_stocks_trading.application.agent_service.research_agents import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from agentic_stocks_trading.domain.tools.analyst_agent_tools import toolkit
from agentic_stocks_trading.infrastructure.llm_providers.llms import quick_thinking_llm

console = Console()
TICKER = "NVDA"
TRADE_DATE = (datetime.date.today() - datetime.timedelta(days=2)).strftime("%Y-%m-%d")
initial_state = AgentState(
    messages=[HumanMessage(content=f"Analyze {TICKER} for trading on {TRADE_DATE}")],
    company_of_interest=TICKER,
    trade_date=TRADE_DATE,
    investment_debate_state=InvestDebateState(
        {
            "history": "",
            "current_response": "",
            "count": 0,
            "bull_history": "",
            "bear_history": "",
            "judge_decision": "",
        }
    ),
    risk_debate_state=RiskDebateState(
        {
            "history": "",
            "latest_speaker": "",
            "current_risky_response": "",
            "current_safe_response": "",
            "current_neutral_response": "",
            "count": 0,
            "risky_history": "",
            "safe_history": "",
            "neutral_history": "",
            "judge_decision": "",
        }
    ),
)

intelligence_gathering_analysts = setup_trading_analysts(quick_thinking_llm, toolkit)

logger.info("Running Market Analyst...")
market_analyst_result = run_analyst(intelligence_gathering_analysts["market_analyst"], initial_state)
initial_state["market_report"] = market_analyst_result.get("market_report", "Failed to generate report.")
console.print("----- Market Analyst Report -----")
console.print(Markdown(initial_state["market_report"]))

logger.info("\nRunning Social Media Analyst...")
social_analyst_result = run_analyst(intelligence_gathering_analysts["social_analyst"], initial_state)
initial_state["sentiment_report"] = social_analyst_result.get("sentiment_report", "Failed to generate report.")
console.print("----- Social Media Analyst Report -----")
console.print(Markdown(initial_state["sentiment_report"]))

# logger.info("\nRunning News Analyst...")
# news_analyst_result = run_analyst(intelligence_gathering_analysts["news_analyst"], initial_state)
# initial_state["news_report"] = news_analyst_result.get("news_report", "Failed to generate report.")
# console.print("----- News Analyst Report -----")
# console.print(Markdown(initial_state["news_report"]))
#
# logger.info("\nRunning Fundamentals Analyst...")
# fundamentals_analyst_result = run_analyst(intelligence_gathering_analysts["fundamentals_analyst"], initial_state)
# initial_state["fundamentals_report"] = fundamentals_analyst_result.get(
#     "fundamentals_report", "Failed to generate report."
# )
# console.print("----- Fundamentals Analyst Report -----")
# console.print(Markdown(initial_state["fundamentals_report"]))

logger.info(initial_state.keys())
