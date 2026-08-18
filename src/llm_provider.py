from __future__ import annotations

"""Shared OpenAI-compatible LLM configuration for generation and RAGAS."""

from dataclasses import dataclass

from config import (
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    GEMINI_EVAL_CHAT_MODEL,
    GEMINI_EVAL_EMBEDDING_MODEL,
    GEMINI_RAGAS_RPM,
    GEMINI_OPENAI_BASE_URL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EVAL_CHAT_MODEL,
    OPENAI_EVAL_EMBEDDING_MODEL,
)


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str
    chat_model: str
    eval_chat_model: str
    embedding_model: str
    base_url: str | None = None


def get_llm_settings() -> LLMSettings | None:
    """Resolve the configured provider without ever exposing its API key."""
    provider = LLM_PROVIDER
    if not provider:
        provider = "openai" if OPENAI_API_KEY else "gemini" if GEMINI_API_KEY else ""

    if provider == "openai" and OPENAI_API_KEY:
        return LLMSettings(
            provider="openai",
            api_key=OPENAI_API_KEY,
            chat_model=OPENAI_CHAT_MODEL,
            eval_chat_model=OPENAI_EVAL_CHAT_MODEL,
            embedding_model=OPENAI_EVAL_EMBEDDING_MODEL,
        )
    if provider == "gemini" and GEMINI_API_KEY:
        return LLMSettings(
            provider="gemini",
            api_key=GEMINI_API_KEY,
            chat_model=GEMINI_CHAT_MODEL,
            eval_chat_model=GEMINI_EVAL_CHAT_MODEL,
            embedding_model=GEMINI_EVAL_EMBEDDING_MODEL,
            base_url=GEMINI_OPENAI_BASE_URL,
        )
    return None


def has_llm_credentials() -> bool:
    return get_llm_settings() is not None


def create_chat_client():
    """Create an OpenAI SDK client and return it with the selected model name."""
    settings = get_llm_settings()
    if settings is None:
        raise RuntimeError(
            "No LLM credentials configured. Set OPENAI_API_KEY or "
            "GEMINI_API_KEY (with LLM_PROVIDER=gemini)."
        )

    from openai import OpenAI

    kwargs = {"api_key": settings.api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return OpenAI(**kwargs), settings.chat_model


def create_ragas_models():
    """Create explicit LangChain LLM/embedding adapters for RAGAS 0.1.x."""
    settings = get_llm_settings()
    if settings is None:
        raise RuntimeError("No LLM credentials configured for RAGAS")

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    common = {"api_key": settings.api_key}
    if settings.base_url:
        common["base_url"] = settings.base_url

    llm_kwargs = {
        "model": settings.eval_chat_model,
        "temperature": 0,
        "max_retries": 3,
        **common,
    }
    if settings.provider == "gemini" and GEMINI_RAGAS_RPM > 0:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        # One shared limiter is attached to the RAGAS ChatOpenAI instance, so
        # concurrent metric workers collectively stay below the account limit.
        llm_kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=GEMINI_RAGAS_RPM / 60.0,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )

    llm = ChatOpenAI(
        **llm_kwargs,
    )
    embedding_kwargs = {
        "model": settings.embedding_model,
        "max_retries": 3,
        **common,
    }
    if settings.provider == "gemini":
        # The tokenizer bundled with langchain-openai does not know Gemini model
        # names; Google's compatibility endpoint handles input limits itself.
        embedding_kwargs.update({
            "tiktoken_enabled": False,
            "check_embedding_ctx_length": False,
        })
    embeddings = OpenAIEmbeddings(**embedding_kwargs)
    return llm, embeddings
