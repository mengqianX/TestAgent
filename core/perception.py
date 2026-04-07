"""视觉感知模块：负责从视频中抽取关键帧。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class ExtractedFrame:
    """单帧抽取结果及调试元数据。"""

    image_path: Path
    timestamp_sec: float
    fps: float
    frame_count: int
    duration_sec: float
    target_frame_index: int
    actual_frame_index: int
    width: int
    height: int


class FrameExtractor:
    """使用 OpenCV 抽取视频关键帧。"""

    def extract_frame(self, video_path: Path, timestamp_sec: float, output_path: Path) -> ExtractedFrame:
        """
        在指定时间戳抽取一帧并保存为图片。

        Args:
            video_path: 输入视频路径。
            timestamp_sec: 目标时间戳（秒）。
            output_path: 输出图片路径。

        Returns:
            ExtractedFrame: 帧文件路径与抽帧元数据。

        Raises:
            FileNotFoundError: 视频文件不存在。
            ValueError: 时间戳非法或超出范围。
            RuntimeError: 抽帧失败。
        """
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        if timestamp_sec < 0:
            raise ValueError(f"时间戳不能为负数: {timestamp_sec}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        try:
            fps: float = cap.get(cv2.CAP_PROP_FPS)
            frame_count_raw: float = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if fps <= 0:
                raise RuntimeError(f"视频 FPS 异常: {fps}")

            frame_count: int = int(round(frame_count_raw)) if frame_count_raw > 0 else 0
            duration_sec: float = frame_count / fps if frame_count > 0 else 0.0
            if duration_sec > 0 and timestamp_sec > duration_sec:
                raise ValueError(
                    f"时间戳超出视频时长: timestamp={timestamp_sec}, duration={duration_sec:.2f}"
                )

            target_frame: int = int(round(timestamp_sec * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            success, frame = cap.read()
            if not success or frame is None:
                raise RuntimeError(f"无法在时间戳 {timestamp_sec} 秒读取帧")
            actual_frame_index: int = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            height: int = int(frame.shape[0])
            width: int = int(frame.shape[1])

            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_ok: bool = cv2.imwrite(str(output_path), frame)
            if not write_ok:
                raise RuntimeError(f"帧写入失败: {output_path}")

            return ExtractedFrame(
                image_path=output_path,
                timestamp_sec=timestamp_sec,
                fps=fps,
                frame_count=frame_count,
                duration_sec=duration_sec,
                target_frame_index=target_frame,
                actual_frame_index=actual_frame_index,
                width=width,
                height=height,
            )
        finally:
            cap.release()

    def extract_frames_by_interval(
        self,
        video_path: Path,
        output_dir: Path,
        prefix: str,
        interval_sec: float = 1.0,
        start_sec: float = 0.0,
        end_sec: float | None = None,
    ) -> list[ExtractedFrame]:
        """
        按固定时间间隔批量抽帧（默认每秒一帧）。

        Args:
            video_path: 输入视频路径。
            output_dir: 输出目录。
            prefix: 输出文件名前缀。
            interval_sec: 抽帧间隔（秒），默认 1 秒。
            start_sec: 起始时间（秒），默认 0。
            end_sec: 结束时间（秒），默认到视频末尾。

        Returns:
            list[ExtractedFrame]: 抽帧结果列表（按时间升序）。
        """
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        if interval_sec <= 0:
            raise ValueError(f"interval_sec 必须大于 0: {interval_sec}")
        if start_sec < 0:
            raise ValueError(f"start_sec 不能小于 0: {start_sec}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        try:
            fps: float = cap.get(cv2.CAP_PROP_FPS)
            frame_count_raw: float = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps <= 0:
                raise RuntimeError(f"视频 FPS 异常: {fps}")

            frame_count: int = int(round(frame_count_raw)) if frame_count_raw > 0 else 0
            duration_sec: float = frame_count / fps if frame_count > 0 else 0.0
            if duration_sec <= 0:
                raise RuntimeError("视频时长异常，无法批量抽帧")

            effective_end_sec = duration_sec if end_sec is None else min(end_sec, duration_sec)
            if effective_end_sec < start_sec:
                raise ValueError(
                    f"end_sec 不能早于 start_sec: start={start_sec}, end={effective_end_sec}"
                )

            output_dir.mkdir(parents=True, exist_ok=True)
            results: list[ExtractedFrame] = []
            current_ts: float = start_sec
            idx: int = 0
            # 使用微小误差防止浮点累积导致最后一帧丢失。
            while current_ts <= effective_end_sec + 1e-9:
                target_frame: int = int(round(current_ts * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                success, frame = cap.read()
                if not success or frame is None:
                    current_ts += interval_sec
                    idx += 1
                    continue

                actual_frame_index: int = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                height: int = int(frame.shape[0])
                width: int = int(frame.shape[1])
                output_path = output_dir / f"{prefix}_t{current_ts:.2f}_idx{idx:04d}.png"

                write_ok: bool = cv2.imwrite(str(output_path), frame)
                if not write_ok:
                    raise RuntimeError(f"帧写入失败: {output_path}")

                results.append(
                    ExtractedFrame(
                        image_path=output_path,
                        timestamp_sec=current_ts,
                        fps=fps,
                        frame_count=frame_count,
                        duration_sec=duration_sec,
                        target_frame_index=target_frame,
                        actual_frame_index=actual_frame_index,
                        width=width,
                        height=height,
                    )
                )
                current_ts += interval_sec
                idx += 1

            return results
        finally:
            cap.release()
