# callbacks/progress_bar.py
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

class ProgressBarCallback(BaseCallback):
    """
    tqdm 프로그레스바: total_timesteps 기준으로 학습 진행률/ETA 표시.
    병렬(VecEnv)에서도 self.model.num_timesteps로 누적 스텝을 정확히 추적.
    """
    def __init__(self, total_timesteps: int, desc: str = "Training", verbose: int = 0):
        super().__init__(verbose)
        self.total = int(total_timesteps)
        self.desc = desc
        self.pbar = None
        self._last = 0

    def _on_training_start(self) -> None:
        self._last = 0
        self.pbar = tqdm(total=self.total, desc=self.desc, unit="step", leave=True)
        return None

    def _on_step(self) -> bool:
        # 현재까지 진행된 timesteps
        cur = int(self.model.num_timesteps)
        delta = cur - self._last
        if delta > 0 and self.pbar is not None:
            self.pbar.update(delta)
            self._last = cur
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            # 혹시 안 닫혔으면 남은 만큼 채우고 종료
            cur = int(self.model.num_timesteps)
            if cur < self.total:
                self.pbar.update(self.total - cur)
            self.pbar.close()
        return None
