import datetime

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from rich.console import Console
from rich.markdown import Markdown

from src.agentic_stocks_trading.application.agent_service.research_agents import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from src.agentic_stocks_trading.domain.prompts.trading_prompts import setup_trading_analysts
from src.agentic_stocks_trading.domain.tools.analyst_agent_tools import toolkit
from src.agentic_stocks_trading.infrastructure.llm_providers.llms import quick_thinking_llm


def run_analyst(analyst_node, initial_state):
    state = initial_state
    # Get all available tools from our toolkit instance.
    all_tools_in_toolkit = [
        getattr(toolkit, name)
        for name in dir(toolkit)
        if callable(getattr(toolkit, name)) and not name.startswith("__")
    ]
    # The ToolNode is a special LangGraph node that executes tool calls.
    tool_node = ToolNode(all_tools_in_toolkit)
    # The ReAct loop can have up to 5 steps of reasoning and tool calls.
    for step in range(5):
        result = analyst_node(state)
        print(result["messages"][-1])
        # The tools_condition checks if the LLM's last message was a tool call.
        if tools_condition(result) == "tools":
            print(f"Step {step + 1}: Calling tools: {[tc['name'] for tc in result['messages'][-1].tool_calls]}")
            state.update(result)
            # Then execute the tools and get tool responses
            tool_result = tool_node.invoke(state)
            print(f"Step {step + 1}: Tools completed")
            # Update state with tool results, but handle messages carefully
            for key, value in tool_result.items():
                if key == "messages":
                    # Append tool response messages to existing messages
                    state["messages"].extend(value)
                else:
                    state[key] = value
        else:
            # If not, the agent is done, so we break the loop.
            state.update(result)
            break
    return state


if __name__ == "__main__":
    console = Console()

    TICKER = "NVDA"
    # Use a recent date for live data fetching
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

    market_analyst_result = run_analyst(intelligence_gathering_analysts["market_analyst"], initial_state)
    initial_state["market_report"] = market_analyst_result.get("market_report", "Failed to generate report.")
    console.print("----- Market Analyst Report -----")
    console.print(Markdown(initial_state["market_report"]))
