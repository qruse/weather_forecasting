"""Aurora 1.5 사전학습 모델 로드.

지역 크롭 입력이면 Swin 의 shifted window 가 동쪽 끝과 서쪽 끝을 이어 붙이지 않게 한다.
(원래 모델은 지구 전체를 가정해서 경도 양 끝을 연결한다)
"""
import torch
from aurora import AuroraV1p5
from aurora.model import swin3d


def disable_lon_wrap():
    f = swin3d.compute_3d_shifted_window_mask
    if getattr(f, "no_wrap", False):
        return

    def no_wrap(*args, **kwargs):
        kwargs["warped"] = False
        return f(*args, **kwargs)

    no_wrap.no_wrap = True
    swin3d.compute_3d_shifted_window_mask = no_wrap


def load_aurora(device=None, regional=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if regional:
        disable_lon_wrap()
    # CPU 에서는 fp16 autocast 가 느리거나 막혀 있어서 끈다
    model = AuroraV1p5(autocast=device == "cuda")
    model.load_checkpoint()
    return model.eval().to(device)
