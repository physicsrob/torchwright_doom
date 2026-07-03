"""onnxruntime-CUDA fp32 sigmoid saturation probe (GPU).

The swiglu op library torchwright is building
(``torchwright/docs/ops_plain_english.md``) makes bit-exactness claims
that assume the deployed inference kernel saturates fp32 sigmoid the
way the CPU kernels do: exactly 1.0 once the input exceeds ~17, exactly
0.0 at -100 (not the representable denormal e^-100), Swish(0) = 0.
Those claims are pinned CPU-side in torchwright's
``tests/docs/test_swish_constants.py`` and torch-CUDA-side in its
``tests/docs/test_swish_saturation_cuda.py``.  This file is the third
kernel: onnxruntime-gpu 1.26.0 under CUDAExecutionProvider — the pair
this repo's Modal image pins and the render runtime deploys.  It lives
here rather than in torchwright because torchwright's Modal test image
carries CPU onnxruntime only (the CPU and GPU builds collide on one
import path), while this suite runs on exactly the deployed pair every
``make test``.

The probe graph is the primitive kernel pattern the gated-MLP emission
will contain — Sigmoid, and Swish as Mul(z, Sigmoid(z)) — run under
default session options (ORT_ENABLE_ALL), so a graph-optimizer fusion
that changed sigmoid numerics would be caught.  Artifact-level
exactness through the full emission is A4's job
(``torchwright/docs/swiglu_step2_plan.md``); this gates the kernel
claims underneath it (A0).

Skipped when torch sees no CUDA device (local CPU-only runs).  On a GPU
box, a failure to create the CUDA session is a test FAILURE, not a
skip — it means the deployed pair is broken.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from onnx import TensorProto, helper

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="ORT-CUDA saturation probe needs a GPU; runs on Modal (make test)",
)

#: torchwright's module hinge-sharpening constant (tests/docs/test_swish_constants.py).
SCALE = 100.0

# Probe points, then a dense sweep of the saturated range.  Index by
# position for the point asserts.
_POINTS = np.array([0.0, 16.0, 17.0, -SCALE, SCALE, 50.0, -50.0], dtype=np.float32)
_SWEEP = np.linspace(17.0, 200.0, 100_001, dtype=np.float32)


def _probe_model() -> bytes:
    """z -> Sigmoid -> sig; Mul(z, sig) -> swish.  Opset 14, matching
    torchwright's exporter."""
    n = len(_POINTS) + len(_SWEEP)
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [n])
    sig = helper.make_tensor_value_info("sig", TensorProto.FLOAT, [n])
    swish = helper.make_tensor_value_info("swish", TensorProto.FLOAT, [n])
    graph = helper.make_graph(
        [
            helper.make_node("Sigmoid", ["z"], ["sig"]),
            helper.make_node("Mul", ["z", "sig"], ["swish"]),
        ],
        "swish_saturation_probe",
        [z],
        [sig, swish],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=10
    )
    return model.SerializeToString()


@pytest.fixture(scope="module")
def probe_outputs():
    import onnxruntime as ort

    sess = ort.InferenceSession(_probe_model(), providers=["CUDAExecutionProvider"])
    # Guard against silent CPU fallback: CUDA EP must have registered
    # first (ORT appends CPUExecutionProvider on its own).
    assert sess.get_providers()[0] == "CUDAExecutionProvider", sess.get_providers()
    z = np.concatenate([_POINTS, _SWEEP])
    sig, swish = sess.run(["sig", "swish"], {"z": z})
    return sig, swish


def test_sigmoid_saturates_to_one_from_17(probe_outputs):
    sig, _ = probe_outputs
    assert sig[2] == 1.0, f"sigmoid(17) = {sig[2]!r}, 1-sig = {1.0 - float(sig[2]):.3e}"
    assert sig[1] < 1.0  # z = 16: the threshold is 17, not lower
    sweep = sig[len(_POINTS) :]
    bad = _SWEEP[sweep != 1.0]
    assert bad.size == 0, (
        f"sigmoid != 1.0 at {bad.size} points in [17, 200]; "
        f"largest offender z={bad.max():.6f}, "
        f"worst 1-sig={(1.0 - sweep[sweep != 1.0]).max():.3e}"
    )


def test_sigmoid_saturates_to_zero_at_minus_scale(probe_outputs):
    sig, swish = probe_outputs
    assert sig[3] == 0.0  # sigma(-100): bit-zero, no denormal leak
    assert swish[3] == 0.0  # a gated select's losing branch gate


def test_swish_fixed_points(probe_outputs):
    _, swish = probe_outputs
    assert swish[0] == 0.0  # Swish(0)
    assert swish[4] == SCALE  # Swish(100) = 100: saturated winning gate


def test_onehot_winner_indicator(probe_outputs):
    _, swish = probe_outputs
    assert swish[5] == 50.0  # hinge(0.5) = Swish(50)/100 = 0.5 exactly
    # hinge(-0.5) leak: e^-50 is representable; bound it (flush-to-zero
    # would be fine too — the budget-relevant direction is the maximum).
    assert abs(swish[6]) <= 1e-19
