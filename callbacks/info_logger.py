# callbacks/info_logger.py
from collections import defaultdict, deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class InfoTensorboardCallback(BaseCallback):
    """
    VecEnv의 infos에서 지정한 키들을 모아 TensorBoard로 기록.
    예) keys=["tau/policy_rms", "tau/vsd_rms", "tau/gravity_rms"]
    """
    def __init__(self, keys, window=1000, prefix="", verbose=0):
        super().__init__(verbose)
        self.keys = list(keys)
        self.window = int(window)
        self.prefix = prefix
        self.buffers = {k: deque(maxlen=self.window) for k in self.keys}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True
        # SubprocVecEnv: infos는 길이 n_envs의 리스트
        for info in infos:
            if not isinstance(info, dict):
                continue
            for k in self.keys:
                v = info.get(k, None)
                if v is not None and np.isfinite(v):
                    self.buffers[k].append(float(v))
        # 주기적으로 기록 (여기서는 매 스텝 기록하되 평균 사용)
        for k, buf in self.buffers.items():
            if len(buf) > 0:
                val = float(np.mean(buf))
                tag = f"{self.prefix}{k}" if self.prefix else k
                self.logger.record(tag, val)
        return True
