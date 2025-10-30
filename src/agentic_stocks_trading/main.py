import datetime

from rich.console import Console
from rich.markdown import Markdown

from agentic_stocks_trading import logger
from agentic_stocks_trading.application.agent_service.nodes_and_runners import (
    create_research_manager,
    create_researcher_node,
    run_analyst,
    setup_trading_analysts,
)
from agentic_stocks_trading.application.agent_service.research_agents import (
    initialize_initial_state,
)
from agentic_stocks_trading.domain.memory.financial_situation import (
    bear_memory,
    bull_memory,
    config,
    invest_judge_memory,
)
from agentic_stocks_trading.domain.prompts.trading_prompts import create_trading_prompt_registry
from agentic_stocks_trading.domain.tools.analyst_agent_tools import toolkit
from agentic_stocks_trading.infrastructure.llm_providers.llms import deep_thinking_llm, quick_thinking_llm

console = Console()

TICKER = "NVDA"
TRADE_DATE = (datetime.date.today() - datetime.timedelta(days=2)).strftime("%Y-%m-%d")

initial_state = initialize_initial_state(TICKER, TRADE_DATE)

registry = create_trading_prompt_registry()

intelligence_gathering_analysts = setup_trading_analysts(quick_thinking_llm, toolkit, registry)

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

logger.info("\nRunning News Analyst...")
news_analyst_result = run_analyst(intelligence_gathering_analysts["news_analyst"], initial_state)
initial_state["news_report"] = news_analyst_result.get("news_report", "Failed to generate report.")
console.print("----- News Analyst Report -----")
console.print(Markdown(initial_state["news_report"]))

logger.info("\nRunning Fundamentals Analyst...")
fundamentals_analyst_result = run_analyst(intelligence_gathering_analysts["fundamentals_analyst"], initial_state)
initial_state["fundamentals_report"] = fundamentals_analyst_result.get(
    "fundamentals_report", "Failed to generate report."
)
console.print("----- Fundamentals Analyst Report -----")
console.print(Markdown(initial_state["fundamentals_report"]))

logger.info(initial_state.keys())

bull_researcher_node = create_researcher_node(quick_thinking_llm, bull_memory, registry, "bull_analyst")
bear_researcher_node = create_researcher_node(quick_thinking_llm, bear_memory, registry, "bear_analyst")

research_manager_node = create_research_manager(deep_thinking_llm, invest_judge_memory)

logger.info("Researcher and Manager agent creation functions are now available.")

current_state = initial_state
for i in range(config.trading.debate.max_debate_rounds):
    logger.info(f"--- Investment Debate Round {i + 1} ---")

    bull_result = bull_researcher_node(current_state)
    current_state["investment_debate_state"] = bull_result["investment_debate_state"]
    console.print("**Bull's Argument:**")
    console.print(Markdown(current_state["investment_debate_state"]["current_response"].replace("Bull Analyst: ", "")))

    bear_result = bear_researcher_node(current_state)
    current_state["investment_debate_state"] = bear_result["investment_debate_state"]
    console.print("**Bear's Rebuttal:**")
    console.print(Markdown(current_state["investment_debate_state"]["current_response"].replace("Bear Analyst: ", "")))
    print("\n")

# After the loops, store the final debate state back into the main initial_state
initial_state["investment_debate_state"] = current_state["investment_debate_state"]

logger.info("Running Research Manager...")
manager_result = research_manager_node(initial_state)
initial_state["investment_plan"] = manager_result["investment_plan"]

console.print("----- Research Manager's Investment Plan -----")
console.print(Markdown(initial_state["investment_plan"]))
