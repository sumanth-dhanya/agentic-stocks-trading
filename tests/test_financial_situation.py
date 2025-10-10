from unittest.mock import Mock, patch

import pytest

from src.agentic_stocks_trading.domain.memory.financial_situation import FinancialSituationMemory


class TestFinancialSituationMemory:
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.trading.llm.backend_url = "http://test-url"
        config.OPENAI_API_KEY = "test-key"
        return config

    @pytest.fixture
    def memory_instance(self, mock_config):
        with (
            patch("src.agentic_stocks_trading.domain.memory.financial_situation.chromadb.Client"),
            patch("src.agentic_stocks_trading.domain.memory.financial_situation.OpenAI"),
        ):
            return FinancialSituationMemory("test", mock_config)

    def test_initialization(self, mock_config):
        with (
            patch("src.agentic_stocks_trading.domain.memory.financial_situation.chromadb.Client") as mock_chroma,
            patch("src.agentic_stocks_trading.domain.memory.financial_situation.OpenAI") as mock_openai,
        ):
            FinancialSituationMemory("test_collection", mock_config)

            mock_openai.assert_called_once_with(
                base_url=mock_config.trading.llm.backend_url, api_key=mock_config.OPENAI_API_KEY
            )
            mock_chroma.assert_called_once()

    def test_add_situations_empty_input(self, memory_instance):
        # Should handle empty input gracefully
        memory_instance.add_situations([])
        # Add assertions based on expected behavior

    def test_get_memories_empty_collection(self, memory_instance):
        memory_instance.situation_collection.count.return_value = 0
        result = memory_instance.get_memories("test situation")
        assert result == []

    @patch.object(FinancialSituationMemory, "get_embedding")
    def test_add_and_retrieve_situations(self, mock_embedding, memory_instance):
        mock_embedding.return_value = [0.1, 0.2, 0.3]  # Mock embedding

        # Mock the count method to return 0 for initial state
        memory_instance.situation_collection.count.return_value = 0

        situations = [("Market is bullish", "Buy stocks"), ("Market is bearish", "Sell stocks")]

        memory_instance.add_situations(situations)
        # Add assertions to verify data was stored correctly

        # Verify that get_embedding was called for each situation
        assert mock_embedding.call_count == 2
        mock_embedding.assert_any_call("Market is bullish")
        mock_embedding.assert_any_call("Market is bearish")

        # Verify that the collection's add method was called with correct parameters
        memory_instance.situation_collection.add.assert_called_once_with(
            documents=["Market is bullish", "Market is bearish"],
            metadatas=[{"recommendation": "Buy stocks"}, {"recommendation": "Sell stocks"}],
            embeddings=[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
            ids=["0", "1"],
        )

    @patch.object(FinancialSituationMemory, "get_embedding")
    def test_add_situations_with_existing_data(self, mock_embedding, memory_instance):
        # Test that IDs are correctly offset when there's existing data
        mock_embedding.return_value = [0.4, 0.5, 0.6]

        # Mock existing data in collection
        memory_instance.situation_collection.count.return_value = 5

        situations = [("New situation", "New advice")]
        memory_instance.add_situations(situations)

        # Verify that IDs start from the existing count
        memory_instance.situation_collection.add.assert_called_once_with(
            documents=["New situation"],
            metadatas=[{"recommendation": "New advice"}],
            embeddings=[[0.4, 0.5, 0.6]],
            ids=["5"],  # Should start from existing count (5)
        )

    def test_get_memories_with_results(self, memory_instance):
        # Mock a non-empty collection
        memory_instance.situation_collection.count.return_value = 3

        # Mock the query response
        mock_query_result = {"metadatas": [[{"recommendation": "Buy stocks"}, {"recommendation": "Hold position"}]]}
        memory_instance.situation_collection.query.return_value = mock_query_result

        # Mock the get_embedding method
        with patch.object(memory_instance, "get_embedding") as mock_embedding:
            mock_embedding.return_value = [0.7, 0.8, 0.9]

            result = memory_instance.get_memories("Current market situation", n_matches=2)

            # Verify get_embedding was called with the current situation
            mock_embedding.assert_called_once_with("Current market situation")

            # Verify collection.query was called with correct parameters
            memory_instance.situation_collection.query.assert_called_once_with(
                query_embeddings=[[0.7, 0.8, 0.9]], n_results=2, include=["metadatas"]
            )

            # Verify the returned result format
            expected_result = [{"recommendation": "Buy stocks"}, {"recommendation": "Hold position"}]
            assert result == expected_result

    def test_get_memories_limits_n_results_to_collection_size(self, memory_instance):
        # Test that n_results is limited to actual collection size
        memory_instance.situation_collection.count.return_value = 2

        mock_query_result = {"metadatas": [[{"recommendation": "Test"}]]}
        memory_instance.situation_collection.query.return_value = mock_query_result

        with patch.object(memory_instance, "get_embedding") as mock_embedding:
            mock_embedding.return_value = [0.1, 0.2, 0.3]

            memory_instance.get_memories("test", n_matches=5)  # Request more than available

            # Should limit n_results to collection size (2)
            memory_instance.situation_collection.query.assert_called_once_with(
                query_embeddings=[[0.1, 0.2, 0.3]],
                n_results=2,  # Limited to collection size
                include=["metadatas"],
            )
