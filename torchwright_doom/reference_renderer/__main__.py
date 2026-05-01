"""CLI entry point: python -m torchwright_doom.reference_renderer [output.png]"""

import argparse

from torchwright_doom.reference_renderer import (
    R_RenderPlayerView,
    RenderConfig,
    box_room,
    generate_trig_table,
    mapdata_from_segments,
    save_png,
)


def main():
    parser = argparse.ArgumentParser(description="Render the box room to a PNG file.")
    parser.add_argument(
        "output", nargs="?", default="box_room.png", help="Output PNG path"
    )
    parser.add_argument("--width", type=int, default=320, help="Screen width in pixels")
    parser.add_argument(
        "--height", type=int, default=200, help="Screen height in pixels"
    )
    parser.add_argument(
        "--fov", type=int, default=64, help="FOV in angle indices (64 ≈ 90°)"
    )
    parser.add_argument("--angle", type=int, default=0, help="Player angle (0-255)")
    args = parser.parse_args()

    config = RenderConfig(
        screen_width=args.width,
        screen_height=args.height,
        fov_columns=args.fov,
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
        player_eye_z=41.0,
    )

    md, textures = mapdata_from_segments(box_room())
    bam = (args.angle << 24) & 0xFFFFFFFF
    frame = R_RenderPlayerView(0.0, 0.0, config.player_eye_z, bam, md, config, textures)
    save_png(frame, args.output)
    print(f"Saved {args.output} ({args.width}x{args.height}, angle={args.angle})")


if __name__ == "__main__":
    main()
