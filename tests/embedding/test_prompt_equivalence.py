"""Plan A / A6: prompt equivalence vs the sandbox.

The real ``build_prompt`` must produce the same prefill token stream as
the sandbox ``get_prefill`` on the same map. The two ``MapData`` schemas
are identical, so we convert a sandbox ``Scene.map_data`` into a real
``MapData`` and run both builders over every test pose, comparing
token-by-token (type name + slot payloads; floats within a tolerance,
ints exact). This exercises every ``ValueRange`` the visible geometry
touches.

Cross-submodule: ``importorskip``\\ s ``doom_sandbox`` (skipped on a
standalone checkout).
"""

from __future__ import annotations

import pytest

from ..sandbox_support import import_sandbox, require_doom_sandbox


# box_room is geometry-only (flat "-"); the sandbox's textured get_prefill
# rejects it. e1m1_subset's flats are compiled into FLAT_ID_BY_NAME.
@pytest.mark.parametrize("fixture_name", ["e1m1_subset"])
def test_build_prompt_matches_sandbox_get_prefill(fixture_name: str) -> None:
    require_doom_sandbox()

    fixtures = import_sandbox("doom_sandbox.fixtures")
    sb_prefill = import_sandbox("doom_sandbox.implementation.prefill")
    from torchwright_doom.prompt.build import build_prompt
    from torchwright_doom.prompt.types import GameState as RealGameState
    from torchwright_doom.prompt.types import MapData as RealMapData

    scene = fixtures.load_fixture(fixture_name)
    real_md = RealMapData(**scene.map_data.model_dump())

    for p, pose in enumerate(scene.test_poses):
        real_state = RealGameState(
            x=pose.x, y=pose.y, angle=pose.angle, viewz=pose.viewz
        )
        real = build_prompt(real_md, real_state)
        sand = sb_prefill.get_prefill(scene, pose)

        assert len(real) == len(
            sand
        ), f"{fixture_name} pose {p}: token count {len(real)} != {len(sand)}"
        for i, (r, s) in enumerate(zip(real, sand)):
            assert r.type.name == s.type.name, (
                f"{fixture_name} pose {p} pos {i}: type "
                f"{r.type.name!r} != {s.type.name!r}"
            )
            assert set(r.values) == set(s.values), (
                f"{fixture_name} pose {p} pos {i} {r.type.name}: slot keys "
                f"{set(r.values)} != {set(s.values)}"
            )
            for k in r.values:
                rv, sv = r.values[k], s.values[k]
                if isinstance(rv, float) or isinstance(sv, float):
                    assert abs(float(rv) - float(sv)) <= 1e-6, (
                        f"{fixture_name} pose {p} pos {i} {r.type.name}.{k}: "
                        f"{rv} != {sv}"
                    )
                else:
                    assert rv == sv, (
                        f"{fixture_name} pose {p} pos {i} {r.type.name}.{k}: "
                        f"{rv} != {sv}"
                    )
