from typing import Final

OPENAI_CHAT_COMPLETIONS_PATH: Final = "/chat/completions"

EMBEDDING_PATH_CANDIDATES: Final[tuple[str, ...]] = (
    "/embeddings",
    "/v1/embeddings",
    "/embed",
    "/v1/embed",
)

RERANK_V1_PATH: Final = "/v1/rerank"
RERANK_V2_PATH: Final = "/v2/rerank"
SCORE_V1_PATH: Final = "/v1/score"

ERR_STREAM_DISABLED: Final = "stream mode is disabled in liteProxy"
ERR_NO_CHAT_MODELS: Final = "no chat model is configured"
ERR_NO_EMBEDDING_MODELS: Final = "no embeddings model is configured"
ERR_NO_RERANK_MODELS: Final = "no reranker model is configured"
