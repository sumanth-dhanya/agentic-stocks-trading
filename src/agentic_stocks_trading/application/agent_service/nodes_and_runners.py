from langgraph.prebuilt import ToolNode, tools_condition

from agentic_stocks_trading.domain.prompts.trading_prompts import TradingPromptRegistry, create_trading_prompt_registry
from src.agentic_stocks_trading.domain.tools.analyst_agent_tools import toolkit


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


# Updated function to create analyst nodes using the registry
def create_analyst_node_from_registry(registry: TradingPromptRegistry, prompt_name: str, llm, toolkit, tools):
    """Create an analyst node using a prompt from the registry"""
    prompt_version = registry.get_active_version(prompt_name)
    if not prompt_version:
        raise ValueError(f"No active version found for prompt: {prompt_name}")

    return prompt_version.create_analyst_node(llm, toolkit, tools)


# usage func
def setup_trading_analysts(quick_thinking_llm, toolkit):
    """Setup all trading analysts using the prompt registry"""
    registry = create_trading_prompt_registry()

    # Create analyst nodes
    market_analyst_node = create_analyst_node_from_registry(
        registry,
        "market_analyst",
        quick_thinking_llm,
        toolkit,
        [toolkit.get_yfinance_data, toolkit.get_technical_indicators],
    )

    social_analyst_node = create_analyst_node_from_registry(
        registry, "social_analyst", quick_thinking_llm, toolkit, [toolkit.get_social_media_sentiment]
    )

    news_analyst_node = create_analyst_node_from_registry(
        registry,
        "news_analyst",
        quick_thinking_llm,
        toolkit,
        [toolkit.get_finnhub_news, toolkit.get_macroeconomic_news],
    )

    fundamentals_analyst_node = create_analyst_node_from_registry(
        registry, "fundamentals_analyst", quick_thinking_llm, toolkit, [toolkit.get_fundamental_analysis]
    )

    return {
        "registry": registry,
        "market_analyst": market_analyst_node,
        "social_analyst": social_analyst_node,
        "news_analyst": news_analyst_node,
        "fundamentals_analyst": fundamentals_analyst_node,
    }


def create_researcher_node(llm, memory, role_prompt, agent_name):
    def researcher_node(state):
        # Combine all reports and debate history for context.
        situation_summary = f"""
        Market Report: {state["market_report"]}
        Sentiment Report: {state["sentiment_report"]}
        News Report: {state["news_report"]}
        Fundamentals Report: {state["fundamentals_report"]}
        """
        past_memories = memory.get_memories(situation_summary)
        past_memory_str = "\n".join([mem["recommendation"] for mem in past_memories])

        prompt = f"""{role_prompt}
        Here is the current state of the analysis:
        {situation_summary}
        Conversation history: {state["investment_debate_state"]["history"]}
        Your opponent's last argument: {state["investment_debate_state"]["current_response"]}
        Reflections from similar past situations: {past_memory_str or "No past memories found."}
        Based on all this information, present your argument conversationally."""

        response = llm.invoke(prompt)
        argument = f"{agent_name}: {response.content}"

        # Update the debate state
        debate_state = state["investment_debate_state"].copy()
        debate_state["history"] += "\n" + argument
        if agent_name == "Bull Analyst":
            debate_state["bull_history"] += "\n" + argument
        else:
            debate_state["bear_history"] += "\n" + argument
        debate_state["current_response"] = argument
        debate_state["count"] += 1
        return {"investment_debate_state": debate_state}

    return researcher_node
