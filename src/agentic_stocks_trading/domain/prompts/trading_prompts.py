from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, field_validator, model_validator


class PromptStatus(str, Enum):
    """Status of a prompt version"""

    DRAFT = "draft"
    TESTING = "testing"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class PromptType(str, Enum):
    """Type/category of prompt"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TEMPLATE = "template"
    LANGCHAIN_CHAT = "langchain_chat"  # New type for LangChain prompts


class TradingRole(str, Enum):
    """Trading-specific analyst roles"""

    MARKET_ANALYST = "market_analyst"
    SOCIAL_ANALYST = "social_analyst"
    NEWS_ANALYST = "news_analyst"
    FUNDAMENTALS_ANALYST = "fundamentals_analyst"
    RISK_ANALYST = "risk_analyst"
    PORTFOLIO_MANAGER = "portfolio_manager"
    QUANTITATIVE_ANALYST = "quantitative_analyst"
    DERIVATIVES_ANALYST = "derivatives_analyst"
    MACRO_ANALYST = "macro_analyst"


class PromptVariable(BaseModel):
    """Definition of a variable that can be used in the prompt"""

    name: str = Field(..., description="Variable name")
    type: str = Field(..., description="Expected data type (str, int, list, etc.)")
    description: str = Field(..., description="Description of what this variable represents")
    required: bool = Field(default=True, description="Whether this variable is required")
    default_value: Any | None = Field(default=None, description="Default value if not provided")
    validation_rules: dict[str, Any] | None = Field(default=None, description="Validation rules for the variable")

    @field_validator("name")
    def validate_variable_name(cls, v):
        if not v.replace("_", "").isalnum():
            raise ValueError("Variable name must be alphanumeric (underscores allowed)")
        return v


class PromptMetadata(BaseModel):
    """Metadata associated with a prompt"""

    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    use_cases: list[str] = Field(default_factory=list, description="Intended use cases")
    model_compatibility: list[str] = Field(default_factory=list, description="Compatible AI models")
    performance_metrics: dict[str, float] | None = Field(default=None, description="Performance metrics")
    cost_estimate: float | None = Field(default=None, description="Estimated cost per execution")
    trading_role: TradingRole | None = Field(default=None, description="Trading role this prompt is designed for")


class TradingPromptVersion(BaseModel):
    """Trading-specific prompt version with LangChain integration"""

    # Core identification
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for this version")
    prompt_name: str = Field(..., description="Name/identifier of the prompt")
    version: str = Field(..., description="Version string (e.g., '1.0.0', '2.1.3')")

    # Prompt content - can be either string template or LangChain components
    content: str | None = Field(default=None, description="Simple prompt text/template")
    system_message: str | None = Field(default=None, description="System message for LangChain ChatPromptTemplate")
    langchain_messages: list[dict[str, Any]] | None = Field(default=None, description="LangChain message structure")

    prompt_type: PromptType = Field(default=PromptType.TEMPLATE, description="Type of prompt")

    # Trading-specific fields
    trading_role: TradingRole | None = Field(default=None, description="Trading role this prompt serves")
    tools_required: list[str] = Field(default_factory=list, description="Required tools for this prompt")
    output_field: str | None = Field(default=None, description="Output field name for LangGraph")

    # Versioning information
    status: PromptStatus = Field(default=PromptStatus.DRAFT, description="Current status")
    parent_version_id: UUID | None = Field(default=None, description="ID of the parent version")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = Field(..., description="User/system that created this version")

    # Documentation and change tracking
    description: str | None = Field(default=None, description="Description of this prompt version")
    changelog: str | None = Field(default=None, description="What changed in this version")
    migration_notes: str | None = Field(default=None, description="Notes for migrating from previous versions")

    # Template variables and validation
    variables: list[PromptVariable] = Field(default_factory=list, description="Variables used in the prompt")
    example_inputs: dict[str, Any] | None = Field(default=None, description="Example variable values")
    expected_output_format: str | None = Field(default=None, description="Description of expected output format")

    # Metadata and configuration
    metadata: PromptMetadata = Field(default_factory=PromptMetadata, description="Additional metadata")
    configuration: dict[str, Any] = Field(default_factory=dict, description="Model-specific configuration")

    # Approval and review
    reviewed_by: list[str] | None = Field(default=None, description="List of reviewers")
    approved_by: str | None = Field(default=None, description="Who approved this version")
    approved_at: datetime | None = Field(default=None, description="When this version was approved")

    # Usage tracking
    usage_count: int = Field(default=0, description="Number of times this version has been used")
    last_used_at: datetime | None = Field(default=None, description="When this version was last used")

    # A/B testing support
    test_group: str | None = Field(default=None, description="Test group identifier for A/B testing")
    rollout_percentage: float = Field(default=100.0, description="Percentage of traffic using this version")

    @field_validator("version")
    def validate_version_format(cls, v):
        """Validate semantic versioning format"""
        import re

        if not re.match(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$", v):
            raise ValueError("Version must follow semantic versioning (e.g., 1.0.0, 2.1.3-beta)")
        return v

    @field_validator("rollout_percentage")
    @classmethod
    def validate_rollout_percentage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Rollout percentage must be between 0 and 100")
        return v

    @model_validator(mode="after")
    def validate_content_and_status(self):
        """Ensure at least one content field is provided and validate status transitions"""
        # Validate content
        if not any([self.content, self.system_message, self.langchain_messages]):
            raise ValueError("At least one of content, system_message, or langchain_messages must be provided")

        # Validate status transitions
        if self.status == PromptStatus.ACTIVE and not self.approved_by:
            raise ValueError("Active prompts must be approved")

        # Auto-set approved_at if approved_by is set but approved_at is not
        if self.approved_by and not self.approved_at:
            self.approved_at = datetime.now(timezone.utc)

        return self

    def create_langchain_prompt(self, tools: list[Any] | None = None) -> ChatPromptTemplate:
        """Create a LangChain ChatPromptTemplate from this prompt version"""
        if self.prompt_type != PromptType.LANGCHAIN_CHAT:
            raise ValueError("This method only works with LANGCHAIN_CHAT prompt types")

        if self.langchain_messages:
            # Use predefined message structure
            messages = []
            for msg in self.langchain_messages:
                if msg["type"] == "system" and msg.get("template"):
                    messages.append(("system", msg["template"]))
                elif msg["type"] == "placeholder":
                    messages.append(MessagesPlaceholder(variable_name=msg["variable_name"]))
                else:
                    messages.append((msg["type"], msg["content"]))
            prompt = ChatPromptTemplate.from_messages(messages)
        elif self.system_message:
            # Use system message with standard structure
            base_system = (
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK; another assistant with different tools"
                " will help where you left off. Execute what you can to make progress."
                " You have access to the following tools: {tool_names}.\n{system_message}"
                " For your reference, the current date is {current_date}. The company we want to look at is {ticker}"
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", base_system),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            # Partial with system message
            prompt = prompt.partial(system_message=self.system_message)

            # Partial with tool names if tools provided
            if tools:
                tool_names = ", ".join([tool.name for tool in tools])
                prompt = prompt.partial(tool_names=tool_names)
        else:
            raise ValueError("No system_message or langchain_messages provided for LangChain prompt creation")

        return prompt

    def create_analyst_node(self, llm, toolkit, tools):
        """Create a LangGraph analyst node from this prompt"""
        if not self.system_message:
            raise ValueError("system_message is required to create an analyst node")

        prompt = self.create_langchain_prompt(tools)

        def analyst_node(state):
            prompt_with_data = prompt.partial(current_date=state["trade_date"], ticker=state["company_of_interest"])
            chain = prompt_with_data | llm.bind_tools(tools)
            result = chain.invoke(state["messages"])

            report = ""
            if not result.tool_calls:
                report = result.content

            return {"messages": [result], self.output_field: report}

        return analyst_node

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}


class TradingPromptRegistry(BaseModel):
    """Registry for managing trading prompt versions"""

    prompts: dict[str, list[TradingPromptVersion]] = Field(default_factory=dict)

    def add_version(self, prompt_version: TradingPromptVersion):
        """Add a new prompt version to the registry"""
        if prompt_version.prompt_name not in self.prompts:
            self.prompts[prompt_version.prompt_name] = []
        self.prompts[prompt_version.prompt_name].append(prompt_version)

    def get_active_version(self, prompt_name: str) -> TradingPromptVersion | None:
        """Get the currently active version of a prompt"""
        if prompt_name not in self.prompts:
            return None

        active_versions = [p for p in self.prompts[prompt_name] if p.status == PromptStatus.ACTIVE]
        if not active_versions:
            return None

        return max(active_versions, key=lambda p: p.created_at)

    def get_prompts_by_role(self, role: TradingRole) -> list[TradingPromptVersion]:
        """Get all active prompts for a specific trading role"""
        result = []
        for prompt_list in self.prompts.values():
            for prompt in prompt_list:
                if prompt.trading_role == role and prompt.status == PromptStatus.ACTIVE:
                    result.append(prompt)
        return result


# Initialize the registry with your existing prompts
def create_trading_prompt_registry():
    """Create and populate the trading prompt registry with existing prompts"""
    registry = TradingPromptRegistry()

    # Market Analyst Prompt
    market_analyst_prompt = TradingPromptVersion(
        prompt_name="market_analyst",
        version="1.0.0",
        system_message="""You are a trading assistant specialized in analyzing financial markets. Your role is to
        select the most relevant technical indicators to analyze a stock's price action, momentum, and volatility.
        You must use your tools to get historical data and then generate a report with your findings,
        including a summary table.""",
        prompt_type=PromptType.LANGCHAIN_CHAT,
        trading_role=TradingRole.MARKET_ANALYST,
        tools_required=["get_yfinance_data", "get_technical_indicators"],
        output_field="market_report",
        created_by="system",
        status=PromptStatus.ACTIVE,
        description="Market analysis prompt focusing on technical indicators",
        approved_by="Sumanth",
        variables=[
            PromptVariable(name="current_date", type="str", description="Current trading date"),
            PromptVariable(name="ticker", type="str", description="Stock ticker symbol"),
            PromptVariable(name="tool_names", type="str", description="Available tool names"),
            PromptVariable(name="messages", type="list", description="Message history"),
        ],
        metadata=PromptMetadata(
            tags=["trading", "technical_analysis", "market"],
            use_cases=["Daily trading analysis", "Technical indicator evaluation"],
            trading_role=TradingRole.MARKET_ANALYST,
        ),
    )

    # Social Media Analyst Prompt
    social_analyst_prompt = TradingPromptVersion(
        prompt_name="social_analyst",
        version="1.0.0",
        system_message="""You are a social media analyst. Your job is to analyze social media posts and public
        sentiment for a specific company over the past week. Use your tools to find relevant discussions and write a
        comprehensive report detailing your analysis, insights, and implications for traders,
        including a summary table.""",
        prompt_type=PromptType.LANGCHAIN_CHAT,
        trading_role=TradingRole.SOCIAL_ANALYST,
        tools_required=["get_social_media_sentiment"],
        output_field="sentiment_report",
        created_by="system",
        status=PromptStatus.ACTIVE,
        description="Social media sentiment analysis prompt",
        approved_by="Sumanth",
        variables=[
            PromptVariable(name="current_date", type="str", description="Current trading date"),
            PromptVariable(name="ticker", type="str", description="Stock ticker symbol"),
            PromptVariable(name="tool_names", type="str", description="Available tool names"),
            PromptVariable(name="messages", type="list", description="Message history"),
        ],
        metadata=PromptMetadata(
            tags=["trading", "sentiment", "social_media"],
            use_cases=["Sentiment analysis", "Social media monitoring"],
            trading_role=TradingRole.SOCIAL_ANALYST,
        ),
    )

    # News Analyst Prompt
    news_analyst_prompt = TradingPromptVersion(
        prompt_name="news_analyst",
        version="1.0.0",
        system_message="""You are a news researcher analyzing recent news and trends over the past week.
         Write a comprehensive report on the current state of the world relevant for trading and macroeconomics.
         Use your tools to be comprehensive and provide detailed analysis, including a summary table.""",
        prompt_type=PromptType.LANGCHAIN_CHAT,
        trading_role=TradingRole.NEWS_ANALYST,
        tools_required=["get_finnhub_news", "get_macroeconomic_news"],
        output_field="news_report",
        created_by="system",
        status=PromptStatus.ACTIVE,
        description="News and macroeconomic analysis prompt",
        approved_by="Sumanth",
        variables=[
            PromptVariable(name="current_date", type="str", description="Current trading date"),
            PromptVariable(name="ticker", type="str", description="Stock ticker symbol"),
            PromptVariable(name="tool_names", type="str", description="Available tool names"),
            PromptVariable(name="messages", type="list", description="Message history"),
        ],
        metadata=PromptMetadata(
            tags=["trading", "news", "macroeconomics"],
            use_cases=["News analysis", "Macroeconomic research"],
            trading_role=TradingRole.NEWS_ANALYST,
        ),
    )

    # Fundamentals Analyst Prompt
    fundamentals_analyst_prompt = TradingPromptVersion(
        prompt_name="fundamentals_analyst",
        version="1.0.0",
        system_message="""You are a researcher analyzing fundamental information about a company.
         Write a comprehensive report on the company's financials, insider sentiment, and transactions to gain a
          full view of its fundamental health, including a summary table.""",
        prompt_type=PromptType.LANGCHAIN_CHAT,
        trading_role=TradingRole.FUNDAMENTALS_ANALYST,
        tools_required=["get_fundamental_analysis"],
        output_field="fundamentals_report",
        created_by="system",
        status=PromptStatus.ACTIVE,
        description="Fundamental analysis prompt for company financials",
        approved_by="Sumanth",
        variables=[
            PromptVariable(name="current_date", type="str", description="Current trading date"),
            PromptVariable(name="ticker", type="str", description="Stock ticker symbol"),
            PromptVariable(name="tool_names", type="str", description="Available tool names"),
            PromptVariable(name="messages", type="list", description="Message history"),
        ],
        metadata=PromptMetadata(
            tags=["trading", "fundamentals", "financials"],
            use_cases=["Fundamental analysis", "Financial health assessment"],
            trading_role=TradingRole.FUNDAMENTALS_ANALYST,
        ),
    )

    # Risk Analyst Prompt (New)
    risk_analyst_prompt = TradingPromptVersion(
        prompt_name="risk_analyst",
        version="1.0.0",
        system_message="""You are a risk management analyst specializing in evaluating portfolio risk,
        market volatility, and position sizing. Analyze the risk metrics, correlation patterns, and
        potential drawdowns. Provide risk-adjusted recommendations and appropriate position sizing strategies,
        including a comprehensive risk assessment table.""",
        prompt_type=PromptType.LANGCHAIN_CHAT,
        trading_role=TradingRole.RISK_ANALYST,
        tools_required=["get_risk_metrics", "get_correlation_analysis", "get_volatility_data"],
        output_field="risk_report",
        created_by="system",
        status=PromptStatus.ACTIVE,
        description="Risk analysis and management prompt",
        approved_by="Sumanth",
        variables=[
            PromptVariable(name="current_date", type="str", description="Current trading date"),
            PromptVariable(name="ticker", type="str", description="Stock ticker symbol"),
            PromptVariable(name="portfolio_value", type="float", description="Total portfolio value", required=False),
            PromptVariable(
                name="risk_tolerance", type="str", description="Risk tolerance level", default_value="moderate"
            ),
        ],
        metadata=PromptMetadata(
            tags=["trading", "risk", "portfolio_management"],
            use_cases=["Risk assessment", "Position sizing", "Portfolio risk management"],
            trading_role=TradingRole.RISK_ANALYST,
        ),
    )

    # Quantitative Analyst Prompt (New)
    quant_analyst_prompt = TradingPromptVersion(
        prompt_name="quantitative_analyst",
        version="1.0.0",
        system_message="""You are a quantitative analyst focusing on statistical analysis, backtesting, and
        algorithmic trading strategies. Analyze historical patterns, statistical relationships, and develop
        quantitative models. Provide backtested results, statistical significance tests, and model performance
        metrics in a detailed analytical report.""",
        prompt_type=PromptType.LANGCHAIN_CHAT,
        trading_role=TradingRole.QUANTITATIVE_ANALYST,
        tools_required=["get_historical_data", "run_backtest", "calculate_statistics", "get_factor_analysis"],
        output_field="quant_report",
        created_by="system",
        status=PromptStatus.ACTIVE,
        description="Quantitative analysis and backtesting prompt",
        approved_by="Sumanth",
        variables=[
            PromptVariable(name="current_date", type="str", description="Current trading date"),
            PromptVariable(name="ticker", type="str", description="Stock ticker symbol"),
            PromptVariable(
                name="lookback_period", type="str", description="Historical data lookback period", default_value="2y"
            ),
            PromptVariable(name="strategy_type", type="str", description="Type of strategy to analyze", required=False),
        ],
        metadata=PromptMetadata(
            tags=["trading", "quantitative", "backtesting", "statistics"],
            use_cases=["Strategy backtesting", "Statistical analysis", "Factor modeling"],
            trading_role=TradingRole.QUANTITATIVE_ANALYST,
        ),
    )

    # Portfolio Manager Prompt (New)
    portfolio_manager_prompt = TradingPromptVersion(
        prompt_name="portfolio_manager",
        version="1.0.0",
        system_message="""You are a portfolio manager responsible for overall portfolio construction, asset
        allocation, and investment decisions. Synthesize information from various analysts to make informed
        portfolio decisions. Focus on diversification, risk-return optimization, and strategic asset allocation.
        Provide specific portfolio recommendations with rationale.""",
        prompt_type=PromptType.LANGCHAIN_CHAT,
        trading_role=TradingRole.PORTFOLIO_MANAGER,
        tools_required=["get_portfolio_analytics", "get_asset_allocation", "get_performance_attribution"],
        output_field="portfolio_report",
        created_by="system",
        status=PromptStatus.ACTIVE,
        description="Portfolio management and asset allocation prompt",
        approved_by="Sumanth",
        variables=[
            PromptVariable(name="current_date", type="str", description="Current trading date"),
            PromptVariable(name="ticker", type="str", description="Stock ticker symbol"),
            PromptVariable(
                name="portfolio_objective", type="str", description="Investment objective", default_value="growth"
            ),
            PromptVariable(
                name="time_horizon", type="str", description="Investment time horizon", default_value="long_term"
            ),
        ],
        metadata=PromptMetadata(
            tags=["trading", "portfolio_management", "asset_allocation"],
            use_cases=["Portfolio construction", "Asset allocation", "Investment decisions"],
            trading_role=TradingRole.PORTFOLIO_MANAGER,
        ),
    )

    # Add all prompts to registry
    for prompt in [
        market_analyst_prompt,
        social_analyst_prompt,
        news_analyst_prompt,
        fundamentals_analyst_prompt,
        risk_analyst_prompt,
        quant_analyst_prompt,
        portfolio_manager_prompt,
    ]:
        registry.add_version(prompt)

    return registry


# Updated function to create analyst nodes using the registry
def create_analyst_node_from_registry(registry: TradingPromptRegistry, prompt_name: str, llm, toolkit, tools):
    """Create an analyst node using a prompt from the registry"""
    prompt_version = registry.get_active_version(prompt_name)
    if not prompt_version:
        raise ValueError(f"No active version found for prompt: {prompt_name}")

    return prompt_version.create_analyst_node(llm, toolkit, tools)


# Example usage function
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


print("Trading prompt registry and analyst creation functions are now available.")
