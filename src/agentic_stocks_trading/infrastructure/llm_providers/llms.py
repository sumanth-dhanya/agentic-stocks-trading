from langchain_openai import ChatOpenAI

from agentic_stocks_trading import get_logger, get_settings

config = get_settings()
llm_config = config.trading.llm
logger = get_logger("llms")

deep_thinking_llm = ChatOpenAI(
    model=llm_config.deep_think_llm, base_url=llm_config.backend_url, temperature=0.1, api_key=config.OPENAI_API_KEY
)

quick_thinking_llm = ChatOpenAI(
    model=llm_config.quick_think_llm, base_url=llm_config.backend_url, temperature=0.1, api_key=config.OPENAI_API_KEY
)


def initialize_llm():
    logger.info("Initializing LLMs")
    return deep_thinking_llm, quick_thinking_llm
