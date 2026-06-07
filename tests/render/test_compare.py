"""Image-level compare stats + dependency-free PNG writing (no GPU)."""

from __future__ import annotations

from torchwright_doom.render import compare


def test_compare_stats():
    gen = {(0, 0): (10, 10, 10), (1, 0): (20, 20, 20), (2, 0): (99, 0, 0)}
    ref = {(0, 0): (10, 10, 10), (1, 0): (21, 21, 21), (3, 0): (5, 5, 5)}
    options = {
        (0, 0): {(10, 10, 10)},
        (1, 0): {(20, 20, 20), (21, 21, 21)},
        (2, 0): {(0, 0, 0)},
    }
    rep = compare.compare(gen, ref, options)
    assert rep.n_gen_pixels == 3
    assert rep.n_ref_pixels == 3
    assert rep.in_both == 2  # (0,0), (1,0)
    assert rep.only_in_gen == 1  # (2,0)
    assert rep.only_in_ref == 1  # (3,0)
    assert rep.exact_color_matches == 1  # only (0,0) is color-identical
    assert rep.gen_with_option == 3  # all three gen pixels have an option set
    # (0,0) in {grey}; (1,0)=20 in {20,21}; (2,0)=99 not in {0} -> 2 of 3
    assert rep.in_option_matches == 2
    assert 0.0 <= rep.in_option_rate <= 1.0
    assert "image compare" in rep.format_short()


def test_write_pngs(tmp_path):
    gen = {(0, 0): (255, 0, 0)}
    ref = {(0, 0): (0, 255, 0), (1, 1): (0, 0, 255)}
    paths = compare.write_pngs(
        gen, ref, tmp_path, options={(0, 0): {(255, 0, 0)}}, scale=4
    )
    assert [p.name for p in paths] == ["generated.png", "reference.png", "diff.png"]
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature
