from typing import Optional
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(case_sensitive=True, env_file=".env")

    # 必需的配置项
    OPENAI_BASE_URL: str
    OPENAI_API_KEY: str
    AUTH_USERNAME: str = "admin"
    AUTH_PASSWORD: str = "admin123"

    # 可选的配置项，提供默认值
    LOG_LEVEL: str = "INFO"  # 默认日志级别
    DEBUG_MODE: Optional[bool] = False  # 默认非调试模式

settings = Settings()
# print(settings)