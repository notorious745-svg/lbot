# ctrader_bridge.py
import json
import requests
from config import MODEL_PATH, SYMBOL
from stable_baselines3 import PPO
from trading_env import TradingEnv

SERVER_URL = "http://localhost:5000/trade"

def run_live():
    env = TradingEnv()
    model = PPO.load(MODEL_PATH, env=env)

    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        payload = {
            "symbol": SYMBOL,
            "action": int(action)
        }
        try:
            requests.post(SERVER_URL, json=payload)
        except Exception as e:
            print(f"Error sending to server: {e}")

        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

if __name__ == "__main__":
    run_live()
