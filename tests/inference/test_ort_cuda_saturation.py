"""onnxruntime-CUDA fp32 sigmoid saturation probe (GPU).

The swiglu op library (``torchwright/docs/ops_plain_english.md``) makes
bit-exactness claims that assume the deployed inference kernel
saturates fp32 sigmoid the way the CPU kernels do: exactly 1.0 once the
input exceeds ~17, exactly 0.0 at -scale (e^-128 sits below fp32's
subnormal floor), Swish(0) = 0.  Those claims are pinned CPU-side in
torchwright's ``tests/docs/test_swish_constants.py`` /
``test_ort_cpu_saturation.py`` and torch-CUDA-side in its
``tests/docs/test_swish_saturation_cuda.py``.  This file is the fourth
kernel: onnxruntime-gpu 1.26.0 under CUDAExecutionProvider — the pair
this repo's Modal image pins and the render runtime deploys.  It lives
here rather than in torchwright because torchwright's Modal test image
carries CPU onnxruntime only (the CPU and GPU builds collide on one
import path), while this suite runs on exactly the deployed pair every
``make test``.

Unlike ORT-CPU (exact 1.0 only from z >= 18), this kernel matches the
torch profile: saturation from 17.  The sweep below re-verifies that on
every run.

The probe graph is the primitive kernel pattern the gated-MLP emission
contains — Sigmoid, and Swish as Mul(z, Sigmoid(z)) — run under default
session options (ORT_ENABLE_ALL), so a graph-optimizer fusion that
changed sigmoid numerics would be caught.  Artifact-level exactness
through the full emission is A4's job
(``torchwright/docs/swiglu_step2_plan.md``); this gates the kernel
claims underneath it (A0).  The no-bias constant-lane pin is the
ORT-CUDA member of the family ``torchwright/docs/no_bias_plan.md``
assigns across kernels — the arithmetic every folded bias rides on in
a ``bias=False`` artifact.

The hinge-sharpening constant and both lane constants are imported
from ``torchwright.ops.const`` — the machine values, not local
literals, so this probe cannot go stale against a torchwright-side
retune (the previous revision pinned a hard-coded 100.0 across the
2026-07-04 move to 128).

Skipped when torch sees no CUDA device (local CPU-only runs).  On a GPU
box, a failure to create the CUDA session is a test FAILURE, not a
skip — it means the deployed pair is broken.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from onnx import TensorProto, helper

from torchwright.ops.const import bias_lane_gate, bias_lane_up, scale

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="ORT-CUDA saturation probe needs a GPU; runs on Modal (make test)",
)

# Probe points, then a dense sweep of the saturated range.  Index by
# position for the point asserts.
_POINTS = np.array(
    [0.0, 16.0, 17.0, -scale, scale, scale / 2, -scale / 2, bias_lane_gate],
    dtype=np.float32,
)
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
    assert sig[3] == 0.0  # sigma(-128): bit-zero, no denormal leak
    assert swish[3] == 0.0  # a gated select's losing branch gate


def test_swish_fixed_points(probe_outputs):
    _, swish = probe_outputs
    assert swish[0] == 0.0  # Swish(0)
    assert swish[4] == scale  # Swish(128) = 128: saturated winning gate


def test_onehot_winner_indicator(probe_outputs):
    _, swish = probe_outputs
    assert swish[5] == scale / 2  # hinge(0.5) = Swish(64)/128 = 0.5 exactly
    # hinge(-0.5) leak: e^-64 is representable (~1.6e-28); bound the
    # hinge form like the torch-CUDA pin (flush-to-zero would be fine
    # too — the budget-relevant direction is the maximum).
    assert abs(swish[6]) / scale <= 1e-27


def test_bias_lane_constants_exact_unit_lane(probe_outputs):
    """The no-bias constant lane (torchwright docs/no_bias_plan.md) on
    the deployed ORT-CUDA kernel: sigma(bias_lane_gate) is exactly 1.0
    (input 32 sits comfortably past the bend), g * sigma(g) lands
    verbatim, and the full lane expression — the GatedMLPSubLayer's
    ``g * sigmoid(g) * u`` — computes exactly 1.0 in fp32, so a constant
    routed through the lane's down-projection row lands verbatim in a
    ``bias=False`` artifact.  Mirrors the torch-CPU / torch-CUDA /
    ORT-CPU pins in torchwright's ``tests/docs/``."""
    sig, swish = probe_outputs
    assert sig[7] == 1.0
    assert swish[7] == bias_lane_gate
    # The x(1/32) fold is a plain IEEE fp32 multiply on both kernels.
    assert np.float32(swish[7]) * np.float32(bias_lane_up) == 1.0
