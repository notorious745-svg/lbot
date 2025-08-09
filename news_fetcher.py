import numpy as np, random
from datetime import datetime

NEXT_EVENT_IN_MIN = (15, 240)  # mock
IMPACT_LEVELS = [0.0, 0.33, 0.66, 1.0]

def get_news_features(now: datetime):
    minutes_to_news = random.randint(*NEXT_EVENT_IN_MIN)
    impact_level    = random.choice(IMPACT_LEVELS)
    minutes_norm = np.clip(minutes_to_news / NEXT_EVENT_IN_MIN[1], 0, 1)
    return minutes_norm, impact_level
