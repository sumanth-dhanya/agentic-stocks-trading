from langgraph.graph import MessagesState
from typing_extensions import TypedDict


# State for the researcher team's debate, acting as a dedicated scratchpad.
class InvestDebateState(TypedDict):
    bull_history: str  # Stores arguments made by the Bull agent.
    bear_history: str  # Stores arguments made by the Bear agent.
    history: str  # The full transcript of the debate.
    current_response: str  # The most recent argument made.
    judge_decision: str  # The manager's final decision.
    count: int  # A counter to track the number of debate rounds.


# State for the risk management team's debate.
class RiskDebateState(TypedDict):
    risky_history: str  # History of the aggressive risk-taker.
    safe_history: str  # History of the conservative agent.
    neutral_history: str  # History of the balanced agent.
    history: str  # Full transcript of the risk discussion.
    latest_speaker: str  # Tracks the last agent to speak.
    current_risky_response: str
    current_safe_response: str
    current_neutral_response: str
    judge_decision: str  # The portfolio manager's final decision.
    count: int  # Counter for risk discussion rounds.


# The main state that will be passed through the entire graph.
# It inherits from MessagesState to include a 'messages' field for chat history.
class AgentState(MessagesState):
    company_of_interest: str  # The stock ticker we are analyzing.
    trade_date: str  # The date for the analysis.
    sender: str  # Tracks which agent last modified the state.
    # Each analyst will populate its own report field.
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str
    # Nested states for the debates.
    investment_debate_state: InvestDebateState
    investment_plan: str  # The plan from the Research Manager.
    trader_investment_plan: str  # The actionable plan from the Trader.
    risk_debate_state: RiskDebateState
    final_trade_decision: str  # The final decision from the Portfolio Manager.
