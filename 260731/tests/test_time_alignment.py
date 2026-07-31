"""无需训练即可运行的时间对齐回归测试。"""
import torch

from ratcgf.config import Config
from ratcgf.dataset import SampleBuilder
from ratcgf.utils.data_loading import load_all


def main():
    cfg = Config()
    cfg.data.data_dir = "__missing_for_synthetic_test__"
    cfg.data.allow_synthetic = True
    bundle = load_all(cfg)
    builder = SampleBuilder(bundle, cfg, torch.device("cpu"))

    assert builder.time_str_at_pos(0) == "2025-01-01 00:00:00"
    assert builder.time_str_at_pos(1) == "2025-01-01 01:00:00"

    # 默认合成 T=240、n_show=97，明确采用末尾对齐：c=0 -> pos=143。
    assert builder._pos(0) == 143
    assert builder.time_str_at_pos(143) == "2025-01-06 23:00:00"

    first = builder.build(builder.valid_centers[0])
    assert first["pos"] == 144
    assert first["time"] == "2025-01-07 00:00:00"
    assert first["target_pos"] == 145
    assert first["target_time"] == "2025-01-07 01:00:00"
    print("TIME_ALIGNMENT_OK")


if __name__ == "__main__":
    main()
