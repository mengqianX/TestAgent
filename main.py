"""vision_gui_agent MVP 入口（JSON 输入 + 统一评估链路）。"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any

from dotenv import load_dotenv  # type: ignore[reportMissingImports]

from core.count_change_detector import ControlBounds, CountChangeDetector
from core.evaluator import VisionEvaluator
from core.pipeline import (
    CountChangeTaskDetector,
    FramePairTaskDetector,
    PipelineOrchestrator,
    ToastTaskDetector,
)
from core.perception import ExtractedFrame, FrameExtractor
from core.preprocessor import GuiPreprocessor
from core.toast_detector import ToastMessageDetector
from utils.logger import setup_logger


@dataclass
class VideoTaskInput:
    """单个任务输入。"""

    task_id: str
    mode: str = "full"
    task_type: str = "general"
    task_type_scope: list[str] | None = None
    video_file: str | None = None
    before_image: str | None = None
    after_image: str | None = None
    sample_interval_sec: float = 1.0
    start_sec: float = 0.0
    end_sec: float | None = None
    control_bounds: ControlBounds | None = None
    expected_count_change: str = "any_change"
    metric_hints: list[str] | None = None
    control_name_hint: str | None = None
    source_base_dir: Path | None = None
    expected_toast_keywords: list[str] | None = None


@dataclass
class RunOptions:
    """运行参数。"""

    output_root: Path | None
    input_json: str | None
    input_file: Path | None


def load_runtime_env(root_dir: Path) -> None:
    """通过 python-dotenv 加载项目根目录 .env。"""
    env_path = root_dir / ".env"
    load_dotenv(dotenv_path=env_path, override=False)


def parse_args() -> RunOptions:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="vision_gui_agent MVP (JSON input)")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None, help="输出根目录")
    parser.add_argument(
        "--input-json",
        dest="input_json",
        type=str,
        default=None,
        help="JSON 字符串输入",
    )
    parser.add_argument("--input-file", dest="input_file", type=str, default=None, help="JSON 文件路径")
    args = parser.parse_args()
    return RunOptions(
        output_root=Path(args.output_dir).expanduser() if args.output_dir else None,
        input_json=args.input_json,
        input_file=Path(args.input_file).expanduser() if args.input_file else None,
    )


def _resolve_expected_change(payload: dict[str, Any]) -> str:
    """
    兼容 expected_count_change 与 expected_passed 两种写法。
    expected_passed=True -> any_change；False -> no_change。
    """
    if payload.get("expected_count_change") is not None:
        return str(payload["expected_count_change"])
    if payload.get("expected_passed") is not None:
        return "any_change" if bool(payload["expected_passed"]) else "no_change"
    return "any_change"


def _parse_bounds(payload: dict[str, Any]) -> ControlBounds | None:
    """
    兼容两种 bounds 输入：
    1) control_bounds: {x,y,width,height}
    2) bounds: [x1,y1,x2,y2]
    """
    if payload.get("control_bounds") is not None:
        return ControlBounds.from_payload(payload["control_bounds"])
    if payload.get("bounds") is not None:
        bounds_raw = payload["bounds"]
        if isinstance(bounds_raw, list):
            return ControlBounds.from_list(bounds_raw)
        if isinstance(bounds_raw, dict):
            return ControlBounds.from_payload(bounds_raw)
        raise ValueError(f"不支持的 bounds 类型: {type(bounds_raw)}")
    return None


def parse_task_from_payload(payload: dict[str, Any], base_dir: Path | None = None) -> VideoTaskInput:
    """将 JSON payload 转为强类型任务对象。"""
    mode = str(payload.get("mode", "full")).strip().lower()
    if mode not in {"full", "targeted"}:
        raise ValueError(f"mode 仅支持 full/targeted，当前为: {mode}")
    task_type = str(payload.get("task_type", payload.get("type", payload.get("prompt_type", "general")))).strip().lower()
    task_type_scope = payload.get("task_type_scope")
    normalized_scope: list[str] | None = None
    if task_type_scope is not None:
        if not isinstance(task_type_scope, list):
            raise ValueError("task_type_scope 必须是数组")
        normalized_scope = [str(x).strip().lower() for x in task_type_scope if str(x).strip()]
        if not normalized_scope:
            normalized_scope = None
    before_image = payload.get("before_image", payload.get("screenshot_a"))
    after_image = payload.get("after_image", payload.get("screenshot_b"))
    missing: list[str] = []
    if mode == "targeted":
        if task_type == "count_change":
            if before_image in ("", None):
                missing.append("before_image|screenshot_a")
            if after_image in ("", None):
                missing.append("after_image|screenshot_b")
            if payload.get("control_bounds") is None and payload.get("bounds") is None:
                missing.append("control_bounds|bounds")
        else:
            if payload.get("video_file") in ("", None):
                missing.append("video_file")
    else:
        has_video = payload.get("video_file") not in ("", None)
        has_count_pair = (
            before_image not in ("", None)
            and after_image not in ("", None)
            and (payload.get("control_bounds") is not None or payload.get("bounds") is not None)
        )
        if not has_video and not has_count_pair:
            missing.append("video_file 或 (before_image+after_image+bounds/control_bounds)")
    if missing:
        raise ValueError(f"mode={mode}, task_type={task_type} 时输入 JSON 缺少必填字段: {missing}")

    return VideoTaskInput(
        task_id=str(payload.get("task_id", "video_task")),
        mode=mode,
        task_type=task_type,
        task_type_scope=normalized_scope,
        video_file=str(payload["video_file"]) if payload.get("video_file") else None,
        before_image=str(before_image) if before_image else None,
        after_image=str(after_image) if after_image else None,
        sample_interval_sec=float(payload.get("sample_interval_sec", 1.0)),
        start_sec=float(payload.get("start_sec", 0.0)),
        end_sec=float(payload["end_sec"]) if payload.get("end_sec") is not None else None,
        control_bounds=_parse_bounds(payload),
        expected_count_change=_resolve_expected_change(payload),
        metric_hints=[str(x) for x in payload.get("metric_hints", [])] if payload.get("metric_hints") else None,
        control_name_hint=(
            str(payload.get("control_name_hint"))
            if payload.get("control_name_hint")
            else (str(payload.get("label")) if payload.get("label") else None)
        ),
        source_base_dir=base_dir,
        expected_toast_keywords=(
            [str(x) for x in payload.get("expected_toast_keywords", [])]
            if payload.get("expected_toast_keywords")
            else ([str(x) for x in payload.get("toast_keywords", [])] if payload.get("toast_keywords") else None)
        ),
    )


def load_task_input(options: RunOptions) -> VideoTaskInput:
    """读取 JSON 输入（优先级：CLI --input-json > CLI --input-file > ENV VGA_INPUT_JSON）。"""
    payload: dict[str, Any]
    if options.input_json:
        payload = json.loads(options.input_json)
        return parse_task_from_payload(payload, base_dir=None)
    if options.input_file:
        payload = json.loads(options.input_file.read_text(encoding="utf-8"))
        return parse_task_from_payload(payload, base_dir=options.input_file.parent)
    env_input = os.getenv("VGA_INPUT_JSON")
    if env_input:
        payload = json.loads(env_input)
        return parse_task_from_payload(payload, base_dir=None)
    raise ValueError("未提供 JSON 输入。请使用 --input-json、--input-file 或 VGA_INPUT_JSON。")


def save_report(report: dict[str, Any], output_dir: Path) -> Path:
    """保存单次执行报告为 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = output_dir / filename
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def run() -> None:
    """执行完整工作流。"""
    run_start = time.perf_counter()
    options = parse_args()
    root_dir = Path(__file__).resolve().parent
    load_runtime_env(root_dir)
    task = load_task_input(options)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root_env = os.getenv("VGA_OUTPUT_DIR")
    output_root = options.output_root
    if output_root is None and output_root_env:
        output_root = Path(output_root_env).expanduser()
    if output_root is None:
        output_root = root_dir / "data" / "runs" / run_id
    if not output_root.is_absolute():
        output_root = (root_dir / output_root).resolve()

    frames_dir = output_root / "frames"
    reports_dir = output_root / "reports"
    debug_dir = output_root / "debug"
    logs_dir = output_root / "logs"
    prompt_log_path = debug_dir / f"vlm_prompts_{run_id}.jsonl"
    prompt_text_path = debug_dir / f"full_prompts_{run_id}.txt"

    logger = setup_logger(log_file_path=logs_dir / "vision_gui_agent.log")
    logger.info("vision_gui_agent 启动")
    logger.info("本次运行输出根目录: %s", output_root)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("未读取到 OPENAI_API_KEY。请检查 .env 或当前终端环境变量。")
    model_name = os.getenv("VISION_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o")
    debug_enabled = os.getenv("VGA_DEBUG", "1").strip() in {"1", "true", "True", "YES", "yes"}
    log_full_data_url = os.getenv("VGA_LOG_FULL_DATA_URL", "0").strip() in {"1", "true", "True", "YES", "yes"}
    enable_preprocess = os.getenv("VGA_ENABLE_PREPROCESS", "1").strip() in {"1", "true", "True", "YES", "yes"}
    preprocess_max_images = int(os.getenv("VGA_PREPROCESS_MAX_EXTRA_IMAGES", "2"))
    toast_top_k_candidates = int(os.getenv("VGA_TOAST_TOP_K_CANDIDATES", "3"))
    toast_min_contour_area = int(os.getenv("VGA_TOAST_MIN_CONTOUR_AREA", "1200"))
    toast_size_target_ratio = float(os.getenv("VGA_TOAST_SCORE_SIZE_TARGET_RATIO", "0.10"))
    toast_size_tolerance = float(os.getenv("VGA_TOAST_SCORE_SIZE_TOLERANCE", "0.10"))
    toast_position_center_ratio = float(os.getenv("VGA_TOAST_SCORE_POSITION_CENTER_RATIO", "0.50"))
    toast_position_tolerance = float(os.getenv("VGA_TOAST_SCORE_POSITION_TOLERANCE", "0.50"))
    toast_weight_size = float(os.getenv("VGA_TOAST_SCORE_WEIGHT_SIZE", "0.55"))
    toast_weight_position = float(os.getenv("VGA_TOAST_SCORE_WEIGHT_POSITION", "0.0"))
    toast_weight_motion = float(os.getenv("VGA_TOAST_SCORE_WEIGHT_MOTION", "0.45"))
    toast_motion_norm_ratio = float(os.getenv("VGA_TOAST_SCORE_MOTION_NORM_RATIO", "0.20"))
    toast_dynamic_penalty_threshold = float(os.getenv("VGA_TOAST_SCORE_DYNAMIC_PENALTY_THRESHOLD", "0.45"))
    toast_dynamic_penalty_scale = float(os.getenv("VGA_TOAST_SCORE_DYNAMIC_PENALTY_SCALE", "1.2"))
    toast_dynamic_penalty_max = float(os.getenv("VGA_TOAST_SCORE_DYNAMIC_PENALTY_MAX", "0.55"))
    toast_candidate_max_area_ratio = float(os.getenv("VGA_TOAST_CANDIDATE_MAX_AREA_RATIO", "0.12"))
    toast_candidate_max_height_ratio = float(os.getenv("VGA_TOAST_CANDIDATE_MAX_HEIGHT_RATIO", "0.30"))
    toast_candidate_min_aspect_ratio = float(os.getenv("VGA_TOAST_CANDIDATE_MIN_ASPECT_RATIO", "1.60"))
    toast_candidate_expand_px = int(os.getenv("VGA_TOAST_CANDIDATE_EXPAND_PX", "10"))
    toast_candidate_full_width_ratio = float(os.getenv("VGA_TOAST_CANDIDATE_FULL_WIDTH_RATIO", "0.96"))
    toast_candidate_edge_touch_px = int(os.getenv("VGA_TOAST_CANDIDATE_EDGE_TOUCH_PX", "3"))
    toast_high_dynamic_threshold = float(os.getenv("VGA_TOAST_HIGH_DYNAMIC_THRESHOLD", "0.20"))
    toast_band_search_ratio = float(os.getenv("VGA_TOAST_BAND_SEARCH_RATIO", "0.22"))
    toast_band_min_area_scale = float(os.getenv("VGA_TOAST_BAND_MIN_AREA_SCALE", "0.40"))
    toast_transition_penalty_threshold = float(os.getenv("VGA_TOAST_TRANSITION_PENALTY_THRESHOLD", "0.08"))
    toast_transition_penalty_scale = float(os.getenv("VGA_TOAST_TRANSITION_PENALTY_SCALE", "1.5"))
    toast_transition_penalty_max = float(os.getenv("VGA_TOAST_TRANSITION_PENALTY_MAX", "0.45"))

    evaluator = VisionEvaluator(
        api_key=api_key,
        model=model_name,
        logger=logger,
        debug=debug_enabled,
        log_full_data_url=log_full_data_url,
        prompt_log_path=prompt_log_path,
        prompt_text_path=prompt_text_path,
    )
    preprocessor = GuiPreprocessor(
        artifact_dir=debug_dir / "preprocess",
        logger=logger,
        max_extra_images=preprocess_max_images,
        toast_min_contour_area=toast_min_contour_area,
        toast_size_target_ratio=toast_size_target_ratio,
        toast_size_tolerance=toast_size_tolerance,
        toast_position_center_ratio=toast_position_center_ratio,
        toast_position_tolerance=toast_position_tolerance,
        toast_weight_size=toast_weight_size,
        toast_weight_position=toast_weight_position,
        toast_weight_motion=toast_weight_motion,
        toast_motion_norm_ratio=toast_motion_norm_ratio,
        toast_dynamic_penalty_threshold=toast_dynamic_penalty_threshold,
        toast_dynamic_penalty_scale=toast_dynamic_penalty_scale,
        toast_dynamic_penalty_max=toast_dynamic_penalty_max,
        toast_candidate_max_area_ratio=toast_candidate_max_area_ratio,
        toast_candidate_max_height_ratio=toast_candidate_max_height_ratio,
        toast_candidate_min_aspect_ratio=toast_candidate_min_aspect_ratio,
        toast_candidate_expand_px=toast_candidate_expand_px,
        toast_candidate_full_width_ratio=toast_candidate_full_width_ratio,
        toast_candidate_edge_touch_px=toast_candidate_edge_touch_px,
        toast_high_dynamic_threshold=toast_high_dynamic_threshold,
        toast_band_search_ratio=toast_band_search_ratio,
        toast_band_min_area_scale=toast_band_min_area_scale,
        toast_transition_penalty_threshold=toast_transition_penalty_threshold,
        toast_transition_penalty_scale=toast_transition_penalty_scale,
        toast_transition_penalty_max=toast_transition_penalty_max,
    )
    count_change_detector = CountChangeDetector(
        evaluator=evaluator,
        logger=logger,
        debug=debug_enabled,
        crops_dir=debug_dir / "count_change_crops",
        preprocessor=preprocessor,
        enable_preprocess=enable_preprocess,
    )
    toast_detector = ToastMessageDetector(
        evaluator=evaluator,
        logger=logger,
        debug=debug_enabled,
        preprocessor=preprocessor,
        enable_preprocess=enable_preprocess,
        top_k_candidates=toast_top_k_candidates,
    )
    extractor = FrameExtractor()

    logger.info(
        "任务输入: task_id=%s, mode=%s, task_type=%s, task_type_scope=%s",
        task.task_id,
        task.mode,
        task.task_type,
        task.task_type_scope,
    )
    logger.info(
        (
            "前处理开关: enable_preprocess=%s, max_extra_images=%s, "
            "toast_top_k_candidates=%s"
        ),
        enable_preprocess,
        preprocess_max_images,
        toast_top_k_candidates,
    )
    logger.info(
        (
            "toast打分配置: min_area=%s, size(target=%.3f,tol=%.3f,w=%.3f), "
            "position(center=%.3f,tol=%.3f,w=%.3f), motion(norm=%.3f,w=%.3f), "
            "penalty(threshold=%.3f,scale=%.3f,max=%.3f), "
            "candidate_filter(max_area=%.3f,max_height=%.3f,min_aspect=%.3f,expand_px=%s,full_width=%.3f,edge_px=%s), "
            "high_dynamic(threshold=%.3f,band_ratio=%.3f,min_area_scale=%.3f), "
            "transition_penalty(threshold=%.3f,scale=%.3f,max=%.3f)"
        ),
        toast_min_contour_area,
        toast_size_target_ratio,
        toast_size_tolerance,
        toast_weight_size,
        toast_position_center_ratio,
        toast_position_tolerance,
        toast_weight_position,
        toast_motion_norm_ratio,
        toast_weight_motion,
        toast_dynamic_penalty_threshold,
        toast_dynamic_penalty_scale,
        toast_dynamic_penalty_max,
        toast_candidate_max_area_ratio,
        toast_candidate_max_height_ratio,
        toast_candidate_min_aspect_ratio,
        toast_candidate_expand_px,
        toast_candidate_full_width_ratio,
        toast_candidate_edge_touch_px,
        toast_high_dynamic_threshold,
        toast_band_search_ratio,
        toast_band_min_area_scale,
        toast_transition_penalty_threshold,
        toast_transition_penalty_scale,
        toast_transition_penalty_max,
    )
    logger.info("Prompt 构造方式: python_builder, requested_type=%s", task.task_type)

    sampled_frames: list[ExtractedFrame] = []
    count_change_pair: tuple[Path, Path] | None = None
    frame_extract_elapsed_ms = 0.0
    if task.before_image and task.after_image:
        before_image_path = Path(task.before_image or "")
        after_image_path = Path(task.after_image or "")
        if not before_image_path.is_absolute():
            base = task.source_base_dir or root_dir
            before_image_path = (base / before_image_path).resolve()
        if not after_image_path.is_absolute():
            base = task.source_base_dir or root_dir
            after_image_path = (base / after_image_path).resolve()
        if not before_image_path.exists() or not after_image_path.exists():
            raise FileNotFoundError(
                f"count_change 输入图片不存在: before={before_image_path}, after={after_image_path}"
            )
        count_change_pair = (before_image_path, after_image_path)
        logger.info("输入图片对: before=%s, after=%s", before_image_path, after_image_path)

    if task.video_file:
        t_frame_start = time.perf_counter()
        video_path = Path(task.video_file)
        if not video_path.is_absolute():
            base = task.source_base_dir or (root_dir / "data")
            video_path = (base / video_path).resolve()
        logger.info("视频输入: video_file=%s, interval=%.2fs", task.video_file, task.sample_interval_sec)
        sampled_frames = extractor.extract_frames_by_interval(
            video_path=video_path,
            output_dir=frames_dir,
            prefix=task.task_id,
            interval_sec=task.sample_interval_sec,
            start_sec=task.start_sec,
            end_sec=task.end_sec,
        )
        frame_extract_elapsed_ms = (time.perf_counter() - t_frame_start) * 1000.0
        logger.info("抽帧完成: 共 %s 帧", len(sampled_frames))

    orchestrator = PipelineOrchestrator(
        detectors=[
            CountChangeTaskDetector(detector=count_change_detector, logger=logger),
            ToastTaskDetector(detector=toast_detector),
            FramePairTaskDetector(evaluator=evaluator, logger=logger),
        ]
    )
    t_pipeline_start = time.perf_counter()
    pipeline_result = orchestrator.run(
        task=task,
        sampled_frames=sampled_frames,
        count_change_pair=count_change_pair,
    )
    pipeline_elapsed_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
    segment_results: list[dict[str, Any]] = pipeline_result.segment_results
    video_bug_detected = pipeline_result.video_bug_detected
    resolved_task_intent = pipeline_result.resolved_task_intent
    token_usage_summary = evaluator.get_token_usage_summary()

    report = {
        "project": "vision_gui_agent",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "task_id": task.task_id,
            "mode": task.mode,
            "task_type": task.task_type,
            "task_type_scope": task.task_type_scope,
            "video_file": task.video_file,
            "before_image": task.before_image,
            "after_image": task.after_image,
            "resolved_task_intent": resolved_task_intent,
            "sample_interval_sec": task.sample_interval_sec,
            "start_sec": task.start_sec,
            "end_sec": task.end_sec,
            "control_bounds": (
                {
                    "x": task.control_bounds.x,
                    "y": task.control_bounds.y,
                    "width": task.control_bounds.width,
                    "height": task.control_bounds.height,
                }
                if task.control_bounds
                else None
            ),
            "expected_count_change": task.expected_count_change,
            "metric_hints": task.metric_hints,
            "control_name_hint": task.control_name_hint,
            "expected_toast_keywords": task.expected_toast_keywords,
        },
        "video_level_result": {
            "bug_detected": video_bug_detected,
            "reason": "任一已执行 detector 判定为 bug，则视频级结果为存在问题。",
            "total_sampled_frames": len(sampled_frames),
            "total_segments": len(segment_results),
        },
        "token_usage_summary": token_usage_summary,
        "debug_artifacts": {
            "output_root": str(output_root),
            "frames_dir": str(frames_dir),
            "logs_dir": str(logs_dir),
            "vlm_prompt_log": str(prompt_log_path),
            "vlm_full_prompt_text": str(prompt_text_path),
            "prompt_builder": "core.prompt_builders.build_prompt_for_type",
            "selected_detector": pipeline_result.selected_detector,
            "detector_runs": pipeline_result.detector_runs,
            "timing": {
                "frame_extract_elapsed_ms": round(frame_extract_elapsed_ms, 2),
                "pipeline_elapsed_ms": round(pipeline_elapsed_ms, 2),
                "total_elapsed_ms": round((time.perf_counter() - run_start) * 1000.0, 2),
            },
        },
        "sampled_frames": (
            [
                {
                    "image_path": str(frame.image_path),
                    "timestamp_sec": frame.timestamp_sec,
                    "target_frame_index": frame.target_frame_index,
                    "actual_frame_index": frame.actual_frame_index,
                    "fps": frame.fps,
                    "width": frame.width,
                    "height": frame.height,
                }
                for frame in sampled_frames
            ]
            if sampled_frames
            else (
                [
                    {"image_path": str(count_change_pair[0]), "timestamp_sec": None},
                    {"image_path": str(count_change_pair[1]), "timestamp_sec": None},
                ]
                if count_change_pair is not None
                else []
            )
        ),
        "segment_results": segment_results,
    }
    report_path = save_report(report=report, output_dir=reports_dir)
    total_elapsed_ms = (time.perf_counter() - run_start) * 1000.0
    logger.info("视频级判定: bug_detected=%s", video_bug_detected)
    logger.info(
        (
            "Token统计: calls=%s, with_usage=%s, without_usage=%s, "
            "prompt_tokens=%s, completion_tokens=%s, total_tokens=%s"
        ),
        token_usage_summary["prompt_call_count"],
        token_usage_summary["calls_with_usage"],
        token_usage_summary["calls_without_usage"],
        token_usage_summary["total_prompt_tokens"],
        token_usage_summary["total_completion_tokens"],
        token_usage_summary["total_tokens"],
    )
    logger.info(
        "运行耗时: total=%.2fms, frame_extract=%.2fms, pipeline=%.2fms",
        total_elapsed_ms,
        frame_extract_elapsed_ms,
        pipeline_elapsed_ms,
    )
    logger.info("测试报告已生成: %s", report_path)


if __name__ == "__main__":
    run()
