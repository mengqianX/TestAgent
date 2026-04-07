"""执行编排模块：统一调度不同任务检测器。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from core.count_change_detector import CountChangeDetector
from core.evaluator import EvaluationResult, VisionEvaluator
from core.perception import ExtractedFrame
from core.prompt_builders import build_prompt_for_type
from core.toast_detector import ToastMessageDetector


@dataclass
class PipelineExecutionResult:
    """任务编排执行结果。"""

    selected_detector: str
    resolved_task_intent: str
    video_bug_detected: bool
    segment_results: list[dict[str, Any]]
    detector_runs: list[dict[str, Any]]


class BaseTaskDetector(ABC):
    """任务检测器抽象基类。"""

    name: str

    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """判断当前检测器是否处理该 task_type。"""

    @abstractmethod
    def is_applicable(
        self,
        task: Any,
        sampled_frames: list[ExtractedFrame],
        count_change_pair: tuple[Path, Path] | None,
    ) -> tuple[bool, str]:
        """判断当前输入是否满足 detector 执行前置条件。"""

    @abstractmethod
    def run(
        self,
        task: Any,
        sampled_frames: list[ExtractedFrame],
        count_change_pair: tuple[Path, Path] | None,
    ) -> PipelineExecutionResult:
        """执行检测并返回统一结构结果。"""


class CountChangeTaskDetector(BaseTaskDetector):
    """count_change 任务检测器。"""

    name = "count_change_detector"

    def __init__(self, detector: CountChangeDetector, logger: logging.Logger | None = None) -> None:
        self.detector = detector
        self.logger = logger or logging.getLogger("vision_gui_agent")

    def can_handle(self, task_type: str) -> bool:
        return task_type == "count_change"

    def is_applicable(
        self,
        task: Any,
        sampled_frames: list[ExtractedFrame],
        count_change_pair: tuple[Path, Path] | None,
    ) -> tuple[bool, str]:
        del sampled_frames
        if task.control_bounds is None:
            return False, "缺少 control_bounds"
        if count_change_pair is None:
            return False, "缺少 before/after 图片对"
        return True, ""

    def run(
        self,
        task: Any,
        sampled_frames: list[ExtractedFrame],
        count_change_pair: tuple[Path, Path] | None,
    ) -> PipelineExecutionResult:
        if task.control_bounds is None:
            raise ValueError("task_type=count_change 时必须提供 control_bounds")
        if count_change_pair is None:
            raise ValueError("count_change 模式缺少输入图片对")

        before_image_path, after_image_path = count_change_pair
        pair_task_id = f"{task.task_id}_pair_0000"
        segment_results: list[dict[str, Any]] = []
        resolved_task_intent = ""
        video_bug_detected = False

        try:
            count_result = self.detector.detect(
                before_image=before_image_path,
                after_image=after_image_path,
                control_bounds=task.control_bounds,
                expected_change=task.expected_count_change,
                metric_hints=task.metric_hints,
                control_name_hint=task.control_name_hint,
                task_id=pair_task_id,
            )
            resolved_task_intent = count_result.task_intent
            video_bug_detected = count_result.bug_detected
            segment_results.append(
                {
                    "segment_index": 0,
                    "segment_task_id": pair_task_id,
                    "before_timestamp_sec": None,
                    "after_timestamp_sec": None,
                    "before_image": str(before_image_path),
                    "after_image": str(after_image_path),
                    "bug_detected": count_result.bug_detected,
                    "reason": count_result.reason,
                    "raw_model_response": count_result.raw_response,
                    "selected_prompt_type": self.name,
                    "semantic_target": count_result.semantic_target,
                    "linked_metric": count_result.linked_metric,
                    "before_value": count_result.before_value,
                    "after_value": count_result.after_value,
                    "value_changed": count_result.value_changed,
                    "change_direction": count_result.change_direction,
                    "expectation_met": count_result.expectation_met,
                    "confidence": count_result.confidence,
                    "timing": count_result.timing,
                    "preprocess_evidence": count_result.preprocess_evidence,
                }
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.exception("图片对评估失败: %s | 错误: %s", pair_task_id, exc)
            segment_results.append(
                {
                    "segment_index": 0,
                    "segment_task_id": pair_task_id,
                    "before_timestamp_sec": None,
                    "after_timestamp_sec": None,
                    "before_image": str(before_image_path),
                    "after_image": str(after_image_path),
                    "error": str(exc),
                }
            )

        return PipelineExecutionResult(
            selected_detector=self.name,
            resolved_task_intent=resolved_task_intent,
            video_bug_detected=video_bug_detected,
            segment_results=segment_results,
            detector_runs=[],
        )


class ToastTaskDetector(BaseTaskDetector):
    """toast 任务检测器。"""

    name = "toast_detector"

    def __init__(self, detector: ToastMessageDetector) -> None:
        self.detector = detector

    def can_handle(self, task_type: str) -> bool:
        return task_type in {"toast", "toast_validation", "toast_content"}

    def is_applicable(
        self,
        task: Any,
        sampled_frames: list[ExtractedFrame],
        count_change_pair: tuple[Path, Path] | None,
    ) -> tuple[bool, str]:
        del task, count_change_pair
        if len(sampled_frames) < 2:
            return False, "抽帧数量不足（至少 2 帧）"
        return True, ""

    def run(
        self,
        task: Any,
        sampled_frames: list[ExtractedFrame],
        count_change_pair: tuple[Path, Path] | None,
    ) -> PipelineExecutionResult:
        del count_change_pair
        toast_result = self.detector.detect(
            sampled_frames=sampled_frames,
            task_id=task.task_id,
            expected_toast_keywords=task.expected_toast_keywords,
        )
        candidate_scores = getattr(toast_result, "candidate_scores", [])
        timing = getattr(toast_result, "timing", None)
        segment_results = [
            {
                "segment_index": 0,
                "segment_task_id": f"{task.task_id}_toast_final",
                "before_timestamp_sec": None,
                "after_timestamp_sec": toast_result.key_frame_timestamp,
                "before_image": None,
                "after_image": str(toast_result.key_frame_path) if toast_result.key_frame_path else None,
                "bug_detected": toast_result.bug_detected,
                "reason": toast_result.reason,
                "raw_model_response": toast_result.raw_response,
                "selected_prompt_type": self.name,
                "toast_text": toast_result.toast_text,
                "action_semantic": toast_result.action_semantic,
                "inferred_expected_toast_text": toast_result.inferred_expected_toast_text,
                "expectation_met": toast_result.expectation_met,
                "confidence": toast_result.confidence,
                "scanned_candidates": toast_result.scanned_candidates,
                "total_candidates": toast_result.total_candidates,
                "evaluated_candidate_indices": toast_result.evaluated_candidate_indices,
                "candidate_scores": candidate_scores,
                "timing": timing,
                "preprocess_evidence": toast_result.preprocess_evidence,
            }
        ]
        return PipelineExecutionResult(
            selected_detector=self.name,
            resolved_task_intent=toast_result.task_intent,
            video_bug_detected=toast_result.bug_detected,
            segment_results=segment_results,
            detector_runs=[],
        )


class FramePairTaskDetector(BaseTaskDetector):
    """通用相邻帧检测器（默认兜底）。"""

    name = "frame_pair_detector"

    def __init__(self, evaluator: VisionEvaluator, logger: logging.Logger | None = None) -> None:
        self.evaluator = evaluator
        self.logger = logger or logging.getLogger("vision_gui_agent")

    def can_handle(self, task_type: str) -> bool:
        del task_type
        return True

    def is_applicable(
        self,
        task: Any,
        sampled_frames: list[ExtractedFrame],
        count_change_pair: tuple[Path, Path] | None,
    ) -> tuple[bool, str]:
        del task, count_change_pair
        if len(sampled_frames) < 2:
            return False, "抽帧数量不足（至少 2 帧）"
        return True, ""

    def run(
        self,
        task: Any,
        sampled_frames: list[ExtractedFrame],
        count_change_pair: tuple[Path, Path] | None,
    ) -> PipelineExecutionResult:
        del count_change_pair
        segment_results: list[dict[str, Any]] = []
        video_bug_detected = False
        resolved_task_intent = ""

        for i in range(len(sampled_frames) - 1):
            before_meta = sampled_frames[i]
            after_meta = sampled_frames[i + 1]
            pair_task_id = f"{task.task_id}_seg_{i:04d}"
            self.logger.info("评估片段 %s: %.2fs -> %.2fs", pair_task_id, before_meta.timestamp_sec, after_meta.timestamp_sec)
            try:
                prompt_context = {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "segment_index": i,
                    "before_timestamp_sec": before_meta.timestamp_sec,
                    "after_timestamp_sec": after_meta.timestamp_sec,
                    "before_image": str(before_meta.image_path),
                    "after_image": str(after_meta.image_path),
                }
                prompt_pack = build_prompt_for_type(task.task_type, prompt_context)
                resolved_task_intent = prompt_pack.task_intent
                t_eval_start = time.perf_counter()
                eval_result: EvaluationResult = self.evaluator.evaluate(
                    before_image=before_meta.image_path,
                    after_image=after_meta.image_path,
                    test_intent=prompt_pack.task_intent,
                    task_id=pair_task_id,
                    system_prompt_override=prompt_pack.system_prompt,
                    user_prompt_override=prompt_pack.user_prompt,
                )
                eval_elapsed_ms = (time.perf_counter() - t_eval_start) * 1000.0
                if eval_result.bug_detected:
                    video_bug_detected = True
                segment_results.append(
                    {
                        "segment_index": i,
                        "segment_task_id": pair_task_id,
                        "before_timestamp_sec": before_meta.timestamp_sec,
                        "after_timestamp_sec": after_meta.timestamp_sec,
                        "before_image": str(before_meta.image_path),
                        "after_image": str(after_meta.image_path),
                        "bug_detected": eval_result.bug_detected,
                        "reason": eval_result.reason,
                        "raw_model_response": eval_result.raw_response,
                        "selected_prompt_type": prompt_pack.selected_prompt_type,
                        "timing": {"vlm_elapsed_ms": round(eval_elapsed_ms, 2)},
                    }
                )
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.exception("片段评估失败: %s | 错误: %s", pair_task_id, exc)
                segment_results.append(
                    {
                        "segment_index": i,
                        "segment_task_id": pair_task_id,
                        "before_timestamp_sec": before_meta.timestamp_sec,
                        "after_timestamp_sec": after_meta.timestamp_sec,
                        "before_image": str(before_meta.image_path),
                        "after_image": str(after_meta.image_path),
                        "error": str(exc),
                    }
                )

        return PipelineExecutionResult(
            selected_detector=self.name,
            resolved_task_intent=resolved_task_intent,
            video_bug_detected=video_bug_detected,
            segment_results=segment_results,
            detector_runs=[],
        )


class PipelineOrchestrator:
    """任务编排器：按 task_type 匹配检测器并执行。"""

    def __init__(self, detectors: list[BaseTaskDetector]) -> None:
        if not detectors:
            raise ValueError("detectors 不能为空")
        self.detectors = detectors
        self.logger = logging.getLogger("vision_gui_agent")

    def run(
        self,
        task: Any,
        sampled_frames: list[ExtractedFrame],
        count_change_pair: tuple[Path, Path] | None,
    ) -> PipelineExecutionResult:
        mode = str(getattr(task, "mode", "full") or "full").strip().lower()
        task_type_hint = str(getattr(task, "task_type", "") or "").strip().lower()
        scope = getattr(task, "task_type_scope", None)
        normalized_scope: set[str] | None = None
        if isinstance(scope, list):
            normalized_scope = {str(x).strip().lower() for x in scope if str(x).strip()}
            if not normalized_scope:
                normalized_scope = None
        is_full_mode = mode != "targeted"
        if is_full_mode:
            candidates = list(self.detectors)
            if normalized_scope is not None:
                candidates = [d for d in candidates if d.name in normalized_scope]
                if not candidates:
                    raise RuntimeError(f"task_type_scope 未匹配任何 detector: {sorted(normalized_scope)}")
            else:
                # 保留 task_type 作为 hint：在 full 模式下优先执行更匹配的 detector。
                prioritized: list[BaseTaskDetector] = []
                remaining: list[BaseTaskDetector] = []
                for detector in candidates:
                    if detector.can_handle(task_type_hint):
                        prioritized.append(detector)
                    else:
                        remaining.append(detector)
                candidates = prioritized + remaining
        else:
            matching = [d for d in self.detectors if d.can_handle(task.task_type)]
            if not matching:
                raise RuntimeError(f"没有可用检测器处理 task_type={task.task_type}")
            # targeted 模式下，明确 task_type 时优先只跑专用 detector，避免误入通用兜底 detector。
            specialized = [d for d in matching if d.name != "frame_pair_detector"]
            if task_type_hint and task_type_hint != "general" and specialized:
                candidates = specialized
            else:
                candidates = matching
        self.logger.info(
            "Pipeline 编排: mode=%s, task_type=%s, candidates=%s",
            mode,
            task_type_hint or task.task_type,
            [d.name for d in candidates],
        )

        all_segments: list[dict[str, Any]] = []
        detector_runs: list[dict[str, Any]] = []
        intents: list[str] = []
        video_bug_detected = False
        executed_names: list[str] = []

        for detector in candidates:
            self.logger.info("Detector 开始评估适用性: %s", detector.name)
            applicable, skip_reason = detector.is_applicable(task, sampled_frames, count_change_pair)
            if not applicable:
                self.logger.info("Detector 跳过: %s, reason=%s", detector.name, skip_reason)
                detector_runs.append(
                    {
                        "detector": detector.name,
                        "status": "skipped",
                        "reason": skip_reason,
                    }
                )
                continue

            try:
                run_start = time.perf_counter()
                self.logger.info("Detector 开始执行: %s", detector.name)
                run_result = detector.run(task, sampled_frames, count_change_pair)
                run_elapsed_ms = (time.perf_counter() - run_start) * 1000.0
                self.logger.info("Detector 执行完成: %s, elapsed=%.2fms", detector.name, run_elapsed_ms)
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.exception("Detector 执行失败: %s, err=%s", detector.name, exc)
                detector_runs.append(
                    {
                        "detector": detector.name,
                        "status": "failed",
                        "reason": str(exc),
                    }
                )
                all_segments.append(
                    {
                        "detector": detector.name,
                        "error": str(exc),
                    }
                )
                continue

            executed_names.append(detector.name)
            if run_result.resolved_task_intent:
                intents.append(run_result.resolved_task_intent)
            if run_result.video_bug_detected:
                video_bug_detected = True

            for segment in run_result.segment_results:
                segment_with_detector = {"detector": detector.name, **segment}
                all_segments.append(segment_with_detector)

            detector_runs.append(
                {
                    "detector": detector.name,
                    "status": "executed",
                    "segment_count": len(run_result.segment_results),
                    "bug_detected": run_result.video_bug_detected,
                    "elapsed_ms": round(run_elapsed_ms, 2),
                }
            )

            if not is_full_mode:
                break

        unique_intents: list[str] = []
        for intent in intents:
            if intent not in unique_intents:
                unique_intents.append(intent)

        selected_detector = (
            "none"
            if not executed_names
            else (executed_names[0] if len(executed_names) == 1 else "multi_detector")
        )
        resolved_intent = " | ".join(unique_intents)

        return PipelineExecutionResult(
            selected_detector=selected_detector,
            resolved_task_intent=resolved_intent,
            video_bug_detected=video_bug_detected,
            segment_results=all_segments,
            detector_runs=detector_runs,
        )
