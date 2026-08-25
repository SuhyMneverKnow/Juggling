"""Preview or record the H1-2 robot and balance-board model."""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "models" / "balance.xml"
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "outputs" / "balance_preview.mp4"
DEFAULT_GIF_PATH = PROJECT_ROOT / "outputs" / "balance_preview.gif"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
INIT_JOINT_POS = {
    "left_hip_pitch_joint": -0.07,
    "left_knee_joint": 0.14,
    "left_ankle_pitch_joint": -0.07,
    "right_hip_pitch_joint": -0.07,
    "right_knee_joint": 0.14,
    "right_ankle_pitch_joint": -0.07,
    "left_elbow_joint": 1.5707963267948966,
    "right_elbow_joint": 1.5707963267948966,
}


def apply_initial_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for joint_name, joint_pos in INIT_JOINT_POS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_id = model.jnt_qposadr[joint_id]
        data.qpos[qpos_id] = joint_pos
    mujoco.mj_forward(model, data)


def configure_camera(cam) -> None:
    cam.lookat[:] = [0.0, 0.0, 0.75]
    cam.distance = 2.8
    cam.azimuth = -135
    cam.elevation = -18


def preview(model: mujoco.MjModel, data: mujoco.MjData, simulate: bool = True) -> None:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_camera(viewer.cam)

        while viewer.is_running():
            step_start = time.time()

            if simulate:
                mujoco.mj_step(model, data)
            viewer.sync()
            target_dt = model.opt.timestep if simulate else 1.0 / 60.0
            sleep_time = target_dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def record_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: Path,
    duration: float,
    fps: int,
    width: int,
    height: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to save MP4 video. Install it with: "
            "conda install -c conda-forge ffmpeg"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_offscreen_size(model, width, height)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    configure_camera(cam)

    frame_count = int(duration * fps)
    frame_dt = 1.0 / fps

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
            assert proc.stdin is not None

            for frame_idx in range(frame_count):
                target_time = frame_idx * frame_dt
                while data.time < target_time:
                    mujoco.mj_step(model, data)

                renderer.update_scene(data, camera=cam)
                proc.stdin.write(renderer.render().tobytes())

            proc.stdin.close()
            return_code = proc.wait()

    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    print(f"Saved {frame_count} frames to {output_path}")


def record_gif(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: Path,
    duration: float,
    fps: int,
    width: int,
    height: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to save GIF with this script. Install it with: "
            "conda install -c conda-forge ffmpeg"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_offscreen_size(model, width, height)
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-filter_complex",
        "[0:v]split[a][b];[a]palettegen[p];[b][p]paletteuse",
        str(output_path),
    ]

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    configure_camera(cam)

    frame_count = int(duration * fps)
    frame_dt = 1.0 / fps

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
            assert proc.stdin is not None

            for frame_idx in range(frame_count):
                target_time = frame_idx * frame_dt
                while data.time < target_time:
                    mujoco.mj_step(model, data)

                renderer.update_scene(data, camera=cam)
                proc.stdin.write(renderer.render().tobytes())

            proc.stdin.close()
            return_code = proc.wait()

    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    print(f"Saved {frame_count} frames to {output_path}")


def ensure_offscreen_size(model: mujoco.MjModel, width: int, height: int) -> None:
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-video", action="store_true", help="Render an MP4 instead of opening the live viewer.")
    parser.add_argument("--save-gif", action="store_true", help="Render a GIF instead of opening the live viewer.")
    parser.add_argument("--static", action="store_true", help="Show the exact initial state without advancing physics.")
    parser.add_argument("--output", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--duration", type=float, default=10.0, help="Video length in seconds.")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_video and args.save_gif:
        raise RuntimeError("Use either --save-video or --save-gif, not both.")
    if args.save_gif and args.output == DEFAULT_VIDEO_PATH:
        args.output = DEFAULT_GIF_PATH

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    apply_initial_pose(model, data)

    if args.save_video:
        record_video(model, data, args.output, args.duration, args.fps, args.width, args.height)
    elif args.save_gif:
        record_gif(model, data, args.output, args.duration, args.fps, args.width, args.height)
    else:
        preview(model, data, simulate=not args.static)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
