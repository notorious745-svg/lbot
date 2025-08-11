import os
from dataclasses import dataclass

@dataclass
class NewsFeature:
    enabled: bool
    source: str
    api_key: str | None

class Config:
    SYMBOL = "XAUUSD"
    TIMEFRAME = "M15"
    NEWS = NewsFeature(
        enabled = os.getenv("LBOT_NEWS_ENABLED", "0") == "1",
        source  = os.getenv("LBOT_NEWS_SOURCE", "none"),
        api_key = os.getenv("LBOT_NEWS_API_KEY")
    )
