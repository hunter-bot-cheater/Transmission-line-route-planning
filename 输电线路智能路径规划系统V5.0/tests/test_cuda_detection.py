import pytest

from transmission_path_planner import config


@pytest.mark.skipif(not config.HAS_TORCH, reason="PyTorch is optional")
def test_cuda_is_not_usable_when_no_device_is_visible(monkeypatch):
    monkeypatch.setattr(config.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(config.torch.cuda, "device_count", lambda: 0)

    assert not config.cuda_is_usable(config.torch)
