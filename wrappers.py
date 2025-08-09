# wrappers.py  — Gymnasium-compatible wrappers
import numpy as np
import pandas as pd
import gymnasium as gym


# --- ทำให้ observation เป็นเวกเตอร์ float32 เสมอ ---
class FeatureVectorObs(gym.ObservationWrapper):
    """
    แปลง obs ให้เป็น 1D float32 vector (รองรับ pandas Series/DataFrame)
    หมายเหตุ: ไม่เปลี่ยน observation_space ระหว่างรัน
    """
    def __init__(self, env):
        super().__init__(env)
        # ถ้า space เป็น Box อยู่แล้ว เราใช้ shape เดิม
        if isinstance(self.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=self.observation_space.shape, dtype=np.float32
            )

    def observation(self, obs):
        if isinstance(obs, (pd.Series, pd.DataFrame)):
            obs = obs.values
        arr = np.asarray(obs, dtype=np.float32).ravel()
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr


# --- shaping reward แบบเบา ๆ ---
class ShapedReward(gym.RewardWrapper):
    """
    ปรับรางวัล: scale + boost ตามเครื่องหมาย (optional)
    ใช้ค่าเริ่มต้นแบบ 'ไม่เปลี่ยนค่า' เพื่อไม่ทำลายสเกลเดิมของ env
    """
    def __init__(self, env, scale: float = 1.0, sign_boost: float = 0.0, clip: float | None = None):
        super().__init__(env)
        self.scale = float(scale)
        self.sign_boost = float(sign_boost)
        self.clip = clip

    def reward(self, reward):
        r = float(reward) * self.scale + self.sign_boost * float(np.sign(reward))
        if self.clip is not None:
            r = float(np.clip(r, -self.clip, self.clip))
        return r


# --- ตัวช่วยเสริม (ถ้าอยากใช้แบบ normalize/scale ง่าย ๆ) ---
class ObservationNormalizeWrapper(gym.ObservationWrapper):
    """normalize obs ต่อ-session (mean/std) แบบง่าย ๆ"""
    def __init__(self, env):
        super().__init__(env)
        if isinstance(self.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=self.observation_space.shape, dtype=np.float32
            )
        self._mean = None
        self._std = None

    def observation(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        if self._mean is None:
            self._mean = float(x.mean())
            self._std = float(x.std() + 1e-9)
        return (x - self._mean) / self._std


class RewardScaleWrapper(gym.RewardWrapper):
    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward):
        return float(reward) * self.scale
