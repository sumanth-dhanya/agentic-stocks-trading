import chromadb
from openai import OpenAI

from agentic_stocks_trading import logger
from src.agentic_stocks_trading.config import get_settings


class FinancialSituationMemory:
    def __init__(self, name, config):
        self.embedding_model = "text-embedding-3-small"
        self.client = OpenAI(base_url=config.trading.llm.backend_url, api_key=config.OPENAI_API_KEY)

        # For persistent storage
        # self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        # self.situation_collection = self.chroma_client.get_or_create_collection(name=name)

        # Use a persistent client for real applications, but in-memory is fine for a notebook.
        self.chroma_client = chromadb.Client(chromadb.config.Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        if not situations_and_advice:
            return
        offset = self.situation_collection.count()
        ids = [str(offset + i) for i, _ in enumerate(situations_and_advice)]
        situations = [s for s, r in situations_and_advice]
        recommendations = [r for s, r in situations_and_advice]
        embeddings = [self.get_embedding(s) for s in situations]
        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in recommendations],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        if self.situation_collection.count() == 0:
            return []
        query_embedding = self.get_embedding(current_situation)
        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_matches, self.situation_collection.count()),
            include=["metadatas"],
        )
        return [{"recommendation": meta["recommendation"]} for meta in results["metadatas"][0]]


logger.info("FinancialSituationMemory class defined.")
config = get_settings()
bull_memory = FinancialSituationMemory("bull_memory", config)
bear_memory = FinancialSituationMemory("bear_memory", config)
trader_memory = FinancialSituationMemory("trader_memory", config)
invest_judge_memory = FinancialSituationMemory("invest_judge_memory", config)
risk_manager_memory = FinancialSituationMemory("risk_manager_memory", config)

logger.info("FinancialSituationMemory instances created for 5 agents.")
