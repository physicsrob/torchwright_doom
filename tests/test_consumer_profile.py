"""Pure arithmetic gates for the consumer-checkpoint memory profiler."""

from __future__ import annotations

import pytest

from scripts.consumer_profile import _dense_fp32_memory


def test_dense_fp32_memory_prices_global_padding_and_cache_growth() -> None:
    report = _dense_fp32_memory(
        n_layers=70,
        max_heads=16,
        max_hidden=8192,
        d=4096,
        d_head=128,
        vocab_size=93378,
        cap_positions=11614,
    )

    assert report["weights_gib"] == pytest.approx(36.42698669433594)
    assert report["cap_kv_gib"] == pytest.approx(12.40509033203125)
    assert report["cache_growth_transient_gib"] == pytest.approx(0.177215576171875)
    assert report["accounted_peak_gib"] == pytest.approx(49.00929260253906)
    assert report["accounted_peak_bytes"] == (
        report["weights_bytes"]
        + report["cap_kv_bytes"]
        + report["cache_growth_transient_bytes"]
    )
