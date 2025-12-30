"""Tests for LLM client pool functionality."""


from delta.llm import ClaudeCodeClient, LLMClientPool, get_classify_client, get_llm_client


class TestLLMClientPool:
    """Tests for LLMClientPool class."""

    def test_singleton_instance(self):
        """Should return same instance on repeated calls."""
        pool1 = LLMClientPool.get_instance()
        pool2 = LLMClientPool.get_instance()
        assert pool1 is pool2

    def test_get_client_returns_cached(self):
        """Should return cached client for same model."""
        pool = LLMClientPool.get_instance()

        client1 = pool.get_client("haiku")
        client2 = pool.get_client("haiku")

        assert client1 is client2

    def test_get_client_different_models(self):
        """Should return different clients for different models."""
        pool = LLMClientPool.get_instance()

        haiku = pool.get_client("haiku")
        sonnet = pool.get_client("sonnet")

        assert haiku is not sonnet

    def test_get_client_default_model(self):
        """Should handle default model (None)."""
        pool = LLMClientPool.get_instance()

        client = pool.get_client(None)
        assert isinstance(client, ClaudeCodeClient)


class TestGetClassifyClient:
    """Tests for get_classify_client function."""

    def test_returns_claude_client(self):
        """Should return a ClaudeCodeClient."""
        client = get_classify_client()
        assert isinstance(client, ClaudeCodeClient)

    def test_uses_pool(self):
        """Should use the client pool for reuse."""
        client1 = get_classify_client("haiku")
        client2 = get_classify_client("haiku")
        assert client1 is client2


class TestGetLlmClient:
    """Tests for get_llm_client function."""

    def test_returns_claude_client(self):
        """Should return a ClaudeCodeClient."""
        client = get_llm_client()
        assert isinstance(client, ClaudeCodeClient)

    def test_uses_pool(self):
        """Should use the client pool for reuse."""
        client1 = get_llm_client("sonnet")
        client2 = get_llm_client("sonnet")
        assert client1 is client2
