"""Scenario list for rollout-based DOOM tests.

Each scenario is one (player state, input) pair.  The rollout fixture
in ``test_rollout.py`` runs ``step_frame`` once per scenario and shares
the resulting :class:`Rollout` across every assertion test.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class Scenario:
    label: str
    px: float
    py: float
    angle: int
    inputs: Dict[str, bool] = field(default_factory=dict)


SCENARIOS = [
    # Center, facing east (angle 0), no movement: every wall renderable
    # but no collision firing.
    Scenario(label="center_east_idle", px=0.0, py=0.0, angle=0),
    # Center, oblique angle (45°): tests off-axis projection — a
    # different sort order than the cardinal scenarios.
    Scenario(label="center_oblique_45", px=0.0, py=0.0, angle=45),
    # Up against the east wall, moving forward into it: HIT_FULL/HIT_X
    # fire on wall 0 (east wall).
    Scenario(
        label="east_wall_collide",
        px=4.9,
        py=0.0,
        angle=0,
        inputs={"forward": True},
    ),
    # Off-center, oblique: stresses the resolved-state and per-wall
    # geometry path with a non-symmetric scene.
    Scenario(label="off_center_3_2_20", px=3.0, py=2.0, angle=20),
    # Strafing into the east wall: HIT_X fires via the strafe-axis ray,
    # not the forward ray.
    Scenario(
        label="strafe_into_east",
        px=4.9,
        py=0.0,
        angle=64,
        inputs={"strafe_right": True},
    ),
    # Near a corner: two walls within ~0.1 unit of the player.
    Scenario(label="corner_position", px=4.9, py=4.9, angle=0),
    # Player almost flush against the east wall: sort_num_t is ~0.1
    # after the central-ray substitution, well above the compare()
    # 0.1-input-unit deadband but tight enough to catch any drift.
    Scenario(label="tiny_distance", px=4.99, py=0.0, angle=0),
]


# Pre-step state for the first frame of `make walkthrough` on the box
# scene (start=(0,0,0); wall-following controller emits forward=True).
# A single frame at walkthrough resolution (d=3072, 120×100) is enough
# to catch prod-config regressions in the compile + step_frame path;
# multi-frame controller coverage costs another full step_frame at this
# resolution per frame and isn't worth the wall-clock.
WALKTHROUGH_SCENARIOS = [
    Scenario(
        label="walkthrough_frame_1",
        px=0.0,
        py=0.0,
        angle=0,
        inputs={"forward": True},
    ),
]
