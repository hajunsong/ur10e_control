# callbacks/save_best_replay.py
import os
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

class EvalWithReplaySave(EvalCallback):
    """
    EvalCallback 확장:
    - best_model.zip 저장 시점에
      · 리플레이 버퍼(.pkl)
      · VecNormalize 통계(.pkl)
      를 함께 저장.
    베스트 갱신 감지는:
      1) super()._on_step() 호출 전후 best_mean_reward 비교
      2) 보조: best_model.zip mtime 변화 감지
    """

    def __init__(
        self, *args,
        save_vecnorm: bool = True,
        replay_name: str = "best_replay_buffer.pkl",
        vecnorm_name: str = "vecnorm.pkl",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_vecnorm = save_vecnorm
        self.replay_name = replay_name
        self.vecnorm_name = vecnorm_name

    def _save_additional(self):
        # 1) 리플레이 버퍼
        if hasattr(self.model, "save_replay_buffer") and (self.model.replay_buffer is not None):
            rb_path = os.path.join(self.best_model_save_path, self.replay_name)
            self.model.save_replay_buffer(rb_path)

        # 2) VecNormalize (학습 env의 통계)
        if self.save_vecnorm:
            env = self.model.get_env()
            if isinstance(env, VecNormalize):
                vn_path = os.path.join(self.best_model_save_path, self.vecnorm_name)
                env.save(vn_path)

    def _on_step(self) -> bool:
        # --- super() 호출 전 상태 스냅샷 ---
        prev_best = self.best_mean_reward
        best_zip = os.path.join(self.best_model_save_path, "best_model.zip")
        prev_mtime = os.path.getmtime(best_zip) if os.path.exists(best_zip) else None

        # --- 원래 평가/저장 수행 ---
        cont = super()._on_step()
        if not cont:
            return False

        # --- 베스트 갱신 감지 ---
        improved = False
        if (self.best_mean_reward is not None) and (prev_best is not None):
            improved = self.best_mean_reward > prev_best

        # 보조: 파일 mtime 변화로도 감지
        cur_mtime = os.path.getmtime(best_zip) if os.path.exists(best_zip) else None
        if (prev_mtime is not None) and (cur_mtime is not None) and (cur_mtime > prev_mtime):
            improved = True

        if improved:
            self._save_additional()

        return True
