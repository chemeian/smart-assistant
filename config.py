"""
Application configuration.
Loads settings from environment variables with sensible defaults.
"""
import os
import json
import base64
import re


def _load_env_file(path=".env"):
    """
    Simple .env file loader (no external dependency).
    Reads KEY=VALUE lines and sets os.environ.
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Remove surrounding quotes if any
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)


# Load .env before any config reads
_env_path = os.path.join(os.path.dirname(__file__), ".env")
_load_env_file(_env_path)


class Config:
    """Central configuration for the smart assistant application."""

    # --- Flask ---
    SECRET_KEY = os.getenv("SECRET_KEY", "smart-assistant-secret-key-change-in-production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))

    # --- CORS ---
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    # --- LLM Provider: DeepSeek ---
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # --- LLM Provider: Tongyi Qianwen ---
    QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
    QWEN_API_BASE = os.getenv("QWEN_API_BASE",
                             "https://dashscope.aliyuncs.com/compatible-mode")
    QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")

    # --- Chat ---
    CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", 20))
    CHAT_TIMEOUT = int(os.getenv("CHAT_TIMEOUT", 60))

    # --- NLP ---
    NLP_LANGUAGE = os.getenv("NLP_LANGUAGE", "chinese")

    # --- Chat File Upload ---
    CHAT_UPLOAD_FOLDER = os.getenv("CHAT_UPLOAD_FOLDER", os.path.join(os.path.dirname(__file__), "data", "chat_uploads"))
    CHAT_MAX_FILE_SIZE_MB = int(os.getenv("CHAT_MAX_FILE_SIZE_MB", 20))
    CHAT_ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
    CHAT_ALLOWED_TEXT_EXT = {"txt", "csv", "json", "md", "xml", "html", "py", "js", "css", "docx"}

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items()
                if not k.startswith("_") and type(v).__name__ in ("str", "int", "bool")}


config = Config()

