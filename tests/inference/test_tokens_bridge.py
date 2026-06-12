"""The sandbox<->real token bridge is total and round-trips (no GPU).

Catches a vocab rename instantly (totality) and proves encode/decode is a fixed
point on the full token stream the renderer actually emits (round-trip).
"""

from __future__ import annotations


from torchwright_doom.inference import tokens_bridge as tb

from ..sandbox_support import import_sandbox, require_doom_sandbox


def _sandbox():
    require_doom_sandbox()
    fixtures = import_sandbox("doom_sandbox.fixtures")
    prefill = import_sandbox("doom_sandbox.implementation.prefill")
    drafter = import_sandbox("doom_sandbox.implementation.reference_drafter")
    setup = import_sandbox("doom_sandbox.implementation.setup")
    return fixtures, prefill, drafter, setup


def test_every_sandbox_type_has_a_real_mirror():
    _, _, _, setup = _sandbox()
    sandbox_names = {t.name for t in setup.VOCAB.types}
    missing = sandbox_names - set(tb._REAL_BY_NAME)
    assert not missing, (
        f"sandbox token types with no torchwright_doom mirror: {sorted(missing)} "
        f"(vocabularies must stay 1:1 — see scripts/vocab_diff.py)"
    )


def test_round_trip_on_full_emitted_stream():
    fixtures, prefill, drafter, _ = _sandbox()
    scene = fixtures.load_fixture("e1m1_subset_textured")
    pose = scene.test_poses[0]
    toks = list(prefill.get_prefill(scene, pose)) + list(
        drafter.expected_ar_tokens(scene, pose)
    )
    seen = set()
    for tok in toks:
        seen.add(tok.type.name)
        row = tb.sandbox_token_to_row(tok)
        back = tb.row_to_sandbox_token(row)
        assert back.type.name == tok.type.name, (tok, back)
        # encode is a fixed point on the quantized value (carriers land on a level)
        assert tb.sandbox_token_to_row(back) == row, (tok, row, back)
    # the stream must exercise the carrier + cursor + pixel + terminal families
    assert {"pixel", "value", "setCursorX", "setCursorDirectionY", "done"} <= seen, seen


def test_rows_to_input_shape():
    t = tb.rows_to_input([3, 7, 11])
    assert tuple(t.shape) == (3, 1)
    assert t.dtype.is_floating_point
    assert [int(x) for x in t[:, 0]] == [3, 7, 11]
