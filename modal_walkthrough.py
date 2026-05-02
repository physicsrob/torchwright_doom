"""Compile the DOOM game graph and generate a walkthrough GIF on Modal.

Usage (via Makefile):
    make walkthrough
    make walkthrough ARGS="--frames 20 --scene e1m1"

Direct usage:
    modal run modal_walkthrough.py
    modal run modal_walkthrough.py --frames 20 --scene e1m1
"""

import sys

import modal

from modal_image import IMAGE

app = modal.App("torchwright-walkthrough", image=IMAGE)


def _scene_data(scene, tex_size):
    """Return ``(subset_md, graph_inputs, start_x, start_y, start_angle,
    max_coord, still, player_eye_z)``.

    ``subset_md`` is the renumbered, mean-centred :class:`MapData`
    (what the reference renderer consumes).  ``graph_inputs`` is the
    transformer-ready slice built from it.  ``still`` is True when
    the scene should be rendered without wall-following motion
    (E1M1's 64-wall subset would walk the player out of loaded
    geometry within a few frames).  ``player_eye_z`` is in raw DOOM
    world units (= sector floor + 41).
    """
    from torchwright_doom.doom.graph_inputs import build_graph_inputs
    from torchwright_doom.doom.subset import (
        DOOM_PLAYER_EYE_HEIGHT,
        build_scene_map_data,
        find_sector_at,
        load_wad_textures_for_subset,
        subset_from_wad,
    )
    from torchwright_doom.reference_renderer.scenes import box_room_textured

    if scene == "box":
        segments, textures = box_room_textured(
            wad_path="doom1.wad",
            tex_size=tex_size,
        )
        subset_md = build_scene_map_data(segments)
        textures_dict = {f"TEX{i}": t for i, t in enumerate(textures)}
        graph_inputs = build_graph_inputs(
            subset_md, textures_dict, max_textures=32, max_bsp_nodes=128,
        )
        return (
            subset_md,
            graph_inputs,
            0.0, 0.0, 0,
            200.0, False, DOOM_PLAYER_EYE_HEIGHT,
        )
    else:  # e1m1
        from torchwright_doom.doom.wad import WADReader

        # Read player-1 spawn (thing type 1) from the WAD.  Angle is in
        # degrees; convert to the renderer's 0-255 scale.
        md = WADReader("doom1.wad").get_map("E1M1")
        spawn = next(t for t in md.things if t.type == 1)
        spawn_x, spawn_y = float(spawn.x), float(spawn.y)
        start_angle = round(spawn.angle / 360 * 256) % 256
        spawn_floor_h = float(md.sectors[find_sector_at(md, spawn_x, spawn_y)].floor_h)
        # See walkthrough.py for the rationale behind max_walls=64.
        subset_md, _orig = subset_from_wad(
            wad_path="doom1.wad",
            map_name="E1M1",
            px=spawn_x,
            py=spawn_y,
            max_walls=64,
            max_bsp_nodes=96,
        )
        textures_dict = load_wad_textures_for_subset(
            "doom1.wad", subset_md, tex_size=tex_size,
        )
        graph_inputs = build_graph_inputs(
            subset_md, textures_dict, max_textures=32, max_bsp_nodes=128,
        )
        return (
            subset_md,
            graph_inputs,
            spawn_x, spawn_y, start_angle,
            4000.0, True, spawn_floor_h + DOOM_PLAYER_EYE_HEIGHT,
        )


def _config(width, height):
    from torchwright_doom.reference_renderer.trig import generate_trig_table
    from torchwright_doom.reference_renderer.types import RenderConfig

    return RenderConfig(
        screen_width=width,
        screen_height=height,
        fov_columns=64,  # ~90° H-FOV, DOOM canonical
        trig_table=generate_trig_table(),
        ceiling_color=(0.2, 0.2, 0.2),
        floor_color=(0.4, 0.4, 0.4),
    )


@app.function(gpu="a100-80gb", cpu=8, timeout=1800)
def generate_transformer(
    scene: str = "box",
    width: int = 320,
    height: int = 200,
    chunk_size: int = 20,
    tex_size: int = 1024,
    frames: int = 10,
    fps: int = 10,
    scale: int = 4,
    d: int = 3072,
    d_hidden: int = 0,
) -> bytes:
    from torchwright_doom.doom.compile import compile_game, step_frame
    from torchwright_doom.doom.walkthrough import generate_walkthrough, save_gif

    config = _config(width, height)
    (
        subset_md, graph_inputs, start_x, start_y, start_angle,
        max_coord, still, eye_z,
    ) = _scene_data(scene, tex_size)
    config.player_eye_z = eye_z
    segments = graph_inputs.segments
    textures = graph_inputs.textures

    print(f"Compiling game graph (walls-as-tokens, {len(segments)} walls)...")
    module = compile_game(
        config,
        textures,
        max_walls=max(8, len(segments)),
        max_coord=max_coord,
        d=d,
        chunk_size=chunk_size,
        device="cuda",
        d_hidden=d_hidden if d_hidden > 0 else None,
    )

    def frame_fn(state, inputs):
        return step_frame(
            module, state, inputs, graph_inputs, config, textures=textures,
        )

    print(f"Generating {frames} transformer frames at {width}x{height}...")
    frame_list = generate_walkthrough(
        segments,
        config,
        frame_fn,
        start_x,
        start_y,
        start_angle,
        total_frames=frames,
        wall_threshold=1.5,
        still=still,
    )

    gif_path = "/tmp/walkthrough.gif"
    save_gif(frame_list, gif_path, fps=fps, scale=scale)

    with open(gif_path, "rb") as f:
        return f.read()


@app.function(cpu=4, timeout=1800)
def generate_reference(
    scene: str = "box",
    width: int = 320,
    height: int = 200,
    tex_size: int = 1024,
    frames: int = 10,
    fps: int = 10,
    scale: int = 4,
) -> bytes:
    from torchwright_doom.doom.game import GameState, update_state
    from torchwright_doom.doom.walkthrough import generate_walkthrough, save_gif
    from torchwright_doom.reference_renderer import R_RenderPlayerView

    config = _config(width, height)
    (
        subset_md, graph_inputs, start_x, start_y, start_angle,
        _, still, eye_z,
    ) = _scene_data(scene, tex_size)
    config.player_eye_z = eye_z
    segments = graph_inputs.segments
    textures = graph_inputs.textures
    # Reference renderer reads texture pixels by name from the
    # subset's sidedefs.  ``graph_inputs.tex_name_to_id`` carries the
    # name→atlas-index map; pair each name with its array.
    ref_textures = {
        name: graph_inputs.textures[i]
        for name, i in graph_inputs.tex_name_to_id.items()
    }
    origin_x, origin_y = graph_inputs.scene_origin

    def frame_fn(state, inputs):
        # Player coords are in world frame; the subset MapData is
        # mean-centred, so shift the player into subset frame for
        # both collision (update_state reads segments) and render.
        shifted = GameState(
            x=state.x - origin_x,
            y=state.y - origin_y,
            angle=state.angle,
            move_speed=state.move_speed,
            turn_speed=state.turn_speed,
        )
        new_shifted = update_state(shifted, inputs, segments, config.trig_table)
        bam = (new_shifted.angle << 24) & 0xFFFFFFFF
        frame = R_RenderPlayerView(
            new_shifted.x,
            new_shifted.y,
            config.player_eye_z,
            bam,
            subset_md,
            config,
            ref_textures,
        )
        new_state = GameState(
            x=new_shifted.x + origin_x,
            y=new_shifted.y + origin_y,
            angle=new_shifted.angle,
            move_speed=new_shifted.move_speed,
            turn_speed=new_shifted.turn_speed,
        )
        return frame, new_state

    print(f"Generating {frames} reference frames at {width}x{height}...")
    frame_list = generate_walkthrough(
        segments,
        config,
        frame_fn,
        start_x,
        start_y,
        start_angle,
        total_frames=frames,
        wall_threshold=1.5,
        still=still,
    )

    gif_path = "/tmp/reference.gif"
    save_gif(frame_list, gif_path, fps=fps, scale=scale)

    with open(gif_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    scene: str = "box",
    width: int = 320,
    height: int = 200,
    chunk_size: int = 20,
    tex_size: int = 1024,
    frames: int = 10,
    fps: int = 10,
    scale: int = 4,
    d: int = 3072,
    d_hidden: int = 0,
):
    # Launch both in parallel
    transformer_call = generate_transformer.spawn(
        scene=scene,
        width=width,
        height=height,
        chunk_size=chunk_size,
        tex_size=tex_size,
        frames=frames,
        fps=fps,
        scale=scale,
        d=d,
        d_hidden=d_hidden,
    )
    reference_call = generate_reference.spawn(
        scene=scene,
        width=width,
        height=height,
        tex_size=tex_size,
        frames=frames,
        fps=fps,
        scale=scale,
    )

    transformer_bytes = transformer_call.get()
    reference_bytes = reference_call.get()

    with open("walkthrough.gif", "wb") as f:
        f.write(transformer_bytes)
    print(f"Saved walkthrough.gif ({len(transformer_bytes)} bytes)")

    with open("reference.gif", "wb") as f:
        f.write(reference_bytes)
    print(f"Saved reference.gif ({len(reference_bytes)} bytes)")
