from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os

class SafetyCallback(BaseCallback):
    """
    Critic loss 발산 시 자동으로 모델 저장 후 학습 중단하는 콜백
    """
    def __init__(self, save_path="checkpoints/safe_ckpt_before_explode.zip", threshold=1e3, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.threshold = threshold

    def _on_step(self) -> bool:
        try:
            # logger 내부 딕셔너리에서 critic loss 가져오기
            if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                critic_loss = self.model.logger.name_to_value.get("train/critic_loss", None)
            else:
                critic_loss = None

            if critic_loss is not None:
                if np.isnan(critic_loss) or abs(critic_loss) > self.threshold:
                    print(f"\n[⚠️ WARN] Critic loss exploding: {critic_loss:.2e}")
                    self.model.save(self.save_path)
                    print(f"[💾 Saved emergency checkpoint] → {self.save_path}")
                    return False  # 학습 중단

        except Exception as e:
            print(f"[SafetyCallback] Error during monitoring: {e}")
        return True
