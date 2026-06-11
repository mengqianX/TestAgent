"""加载异常检测模块：CV-first + VLM-fallback。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2

from core.evaluator import VisionEvaluator
from core.prompt_builders import build_loading_failure_probe_prompt, build_prompt_for_type


@dataclass
class LoadingDetectionResult:
    """loading 检测结果。"""

    bug_detected: bool
    task_intent: str
    reason: str
    decision_basis: str
    anomaly_type: str
    anomaly_types: list[str]
    raw_response: str
    decision_source: str
    cv_metrics: dict[str, Any]
    timing: dict[str, float] | None = None


class LoadingDetector:
    """优先使用 CV 信号判定加载异常，不确定时回退到 VLM。"""

    def __init__(
        self,
        evaluator: VisionEvaluator,
        logger: logging.Logger | None = None,
        debug: bool = False,
        failure_probe_top_k: int = 5,
        failure_probe_min_cv_score: float = 0.02,
        failure_probe_force_keep: int = 1,
        cv_static_mean_threshold: float = 0.006,
        cv_static_max_threshold: float = 0.015,
        cv_spinner_mean_upper: float = 0.03,
        cv_spinner_std_upper: float = 0.008,
        cv_clear_progress_max_threshold: float = 0.08,
    ) -> None:
        self.evaluator = evaluator
        self.logger = logger or logging.getLogger("vision_gui_agent")
        self.debug = debug
        self.failure_probe_top_k = max(1, int(failure_probe_top_k))
        self.failure_probe_min_cv_score = max(0.0, float(failure_probe_min_cv_score))
        self.failure_probe_force_keep = max(1, int(failure_probe_force_keep))
        self.cv_static_mean_threshold = cv_static_mean_threshold
        self.cv_static_max_threshold = cv_static_max_threshold
        self.cv_spinner_mean_upper = cv_spinner_mean_upper
        self.cv_spinner_std_upper = cv_spinner_std_upper
        self.cv_clear_progress_max_threshold = cv_clear_progress_max_threshold

    @staticmethod
    def _read_gray_image(path: Path) -> Any:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _calc_change_ratio(gray_a: Any, gray_b: Any, threshold: int = 22) -> float:
        diff = cv2.absdiff(gray_a, gray_b)
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        changed_pixels = int(cv2.countNonZero(binary))
        total_pixels = max(1, int(binary.shape[0] * binary.shape[1]))
        return changed_pixels / total_pixels

    @staticmethod
    def _build_stats(ratios: list[float]) -> dict[str, float]:
        if not ratios:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
        mean = sum(ratios) / len(ratios)
        variance = sum((x - mean) ** 2 for x in ratios) / len(ratios)
        std = variance**0.5
        return {"mean": mean, "max": max(ratios), "min": min(ratios), "std": std}

    @staticmethod
    def _calc_screen_stats(gray: Any) -> dict[str, float]:
        total = max(1, int(gray.shape[0] * gray.shape[1]))
        black_pixels = int((gray <= 16).sum())
        white_pixels = int((gray >= 239).sum())
        mean_val = float(gray.mean())
        std_val = float(gray.std())
        return {
            "mean_luma": mean_val,
            "std_luma": std_val,
            "black_ratio": black_pixels / total,
            "white_ratio": white_pixels / total,
        }

    def _detect_black_white_screen(self, sampled_frames: list[Any]) -> tuple[str | None, str, dict[str, Any]]:
        """
        识别黑/白屏异常。优先检查末帧，避免把启动页误判为异常。
        """
        last_gray = self._read_gray_image(sampled_frames[-1].image_path)
        stats = self._calc_screen_stats(last_gray)
        black_ratio = float(stats["black_ratio"])
        white_ratio = float(stats["white_ratio"])
        std_luma = float(stats["std_luma"])
        if black_ratio >= 0.92 and std_luma <= 12.0:
            return "black_screen", "CV判定末帧为近纯黑画面，疑似黑屏异常。", stats
        if white_ratio >= 0.92 and std_luma <= 12.0:
            return "white_screen", "CV判定末帧为近纯白画面，疑似白屏异常。", stats
        return None, "", stats

    def _cv_decide(self, ratios: list[float]) -> tuple[bool | None, str, str, dict[str, Any]]:
        stats = self._build_stats(ratios)
        if not ratios:
            return None, "CV证据不足：缺少相邻帧变化数据。", "unknown", {"frame_change_ratios": ratios, **stats}

        mean_ratio = float(stats["mean"])
        max_ratio = float(stats["max"])
        std_ratio = float(stats["std"])

        if max_ratio >= self.cv_clear_progress_max_threshold:
            return (
                False,
                "CV判定存在明显页面变化，未见持续加载卡死特征。",
                "none",
                {"frame_change_ratios": ratios, **stats},
            )

        if mean_ratio <= self.cv_static_mean_threshold and max_ratio <= self.cv_static_max_threshold:
            return (
                True,
                "CV判定页面长时间近乎静止，疑似加载无反馈/卡住。",
                "no_response",
                {"frame_change_ratios": ratios, **stats},
            )

        if mean_ratio <= self.cv_spinner_mean_upper and std_ratio <= self.cv_spinner_std_upper:
            return (
                True,
                "CV判定持续小幅规律变化，疑似加载指示器长期存在。",
                "long_loading",
                {"frame_change_ratios": ratios, **stats},
            )

        return (
            None,
            "CV判定不确定：变化特征介于静止与明显进展之间，转交VLM语义判定。",
            "unknown",
            {"frame_change_ratios": ratios, **stats},
        )

    @staticmethod
    def _unique_keep_order(items: list[str]) -> list[str]:
        out: list[str] = []
        for item in items:
            if item and item not in out:
                out.append(item)
        return out

    @staticmethod
    def _pick_primary_anomaly(anomaly_types: list[str]) -> str:
        if not anomaly_types:
            return "none"
        priority = [
            "black_screen",
            "white_screen",
            "load_failed",
            "no_response",
            "long_loading",
            "unknown",
            "none",
        ]
        for p in priority:
            if p in anomaly_types:
                return p
        return anomaly_types[0]

    def _run_failure_text_probe(
        self,
        before_image: Path,
        center_image: Path,
        after_image: Path,
        before_ts: float,
        center_ts: float,
        after_ts: float,
        task_id: str,
        candidate_index: int,
    ) -> dict[str, Any]:
        """
        加载失败探测：复用 VLM 识别失败文案/失败弹窗信号。
        """
        prompt_pack = build_loading_failure_probe_prompt(
            context={
                "before_timestamp_sec": before_ts,
                "after_timestamp_sec": center_ts,
            }
        )
        probe_result = self.evaluator.evaluate_json(
            before_image=before_image,
            after_image=center_image,
            task_id=f"{task_id}_failure_probe_{candidate_index:04d}",
            system_prompt=prompt_pack.system_prompt,
            user_prompt=prompt_pack.user_prompt,
            required_fields={
                "load_failed": bool,
                "failure_type": str,
                "evidence_text": str,
                "reason": str,
            },
            extra_image_paths=[after_image],
            image_role_labels=[
                "图1:候选前帧（动作后早期状态）",
                "图2:候选帧（重点检测失败提示/弹窗）",
                "图3:候选后帧（验证提示是否短暂出现）",
            ],
        )
        parsed = probe_result.parsed_json
        confidence_raw = parsed.get("confidence")
        confidence = None
        if isinstance(confidence_raw, (int, float)):
            confidence = float(confidence_raw)
        return {
            "candidate_index": candidate_index,
            "before_ts": before_ts,
            "center_ts": center_ts,
            "after_ts": after_ts,
            "load_failed": bool(parsed["load_failed"]),
            "failure_type": str(parsed.get("failure_type", "unknown")).strip().lower() or "unknown",
            "evidence_text": str(parsed.get("evidence_text", "")),
            "reason": str(parsed.get("reason", "")),
            "confidence": confidence,
            "raw_response": probe_result.raw_response,
        }

    def _select_failure_probe_indices(self, sampled_frames: list[Any]) -> list[int]:
        total = len(sampled_frames)
        if total <= self.failure_probe_top_k:
            return list(range(total))
        # 先保留首/中/尾，再用等间距补齐，兼顾瞬时提示与全段覆盖。
        anchors = {0, total // 2, total - 1}
        if self.failure_probe_top_k <= len(anchors):
            return sorted(list(anchors))[: self.failure_probe_top_k]
        slots = self.failure_probe_top_k - len(anchors)
        for i in range(1, slots + 1):
            idx = int(round(i * (total - 1) / (slots + 1)))
            anchors.add(max(0, min(total - 1, idx)))
        selected = sorted(list(anchors))
        if len(selected) > self.failure_probe_top_k:
            selected = selected[: self.failure_probe_top_k]
        return selected

    def _scan_failure_prompt_candidates(
        self,
        sampled_frames: list[Any],
        task_id: str,
        candidate_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        selected_indices = candidate_indices or self._select_failure_probe_indices(sampled_frames)
        gray_cache: dict[int, Any] = {}

        def _gray(idx: int) -> Any:
            if idx not in gray_cache:
                gray_cache[idx] = self._read_gray_image(sampled_frames[idx].image_path)
            return gray_cache[idx]

        # 轻量CV预筛：优先挑“中间帧相对前后有明显变化”的候选，减少VLM调用。
        prefilter_scores: list[dict[str, Any]] = []
        for idx in selected_indices:
            before_idx = max(0, idx - 1)
            after_idx = min(len(sampled_frames) - 1, idx + 1)
            before_center = self._calc_change_ratio(_gray(before_idx), _gray(idx))
            center_after = self._calc_change_ratio(_gray(idx), _gray(after_idx))
            # 既考虑瞬时出现（单侧变化大），也考虑持续提示（双侧都有变化）。
            cv_score = max(before_center, center_after) + 0.5 * min(before_center, center_after)
            prefilter_scores.append(
                {
                    "index": idx,
                    "before_center_ratio": round(before_center, 4),
                    "center_after_ratio": round(center_after, 4),
                    "cv_score": round(cv_score, 4),
                }
            )

        prefilter_scores.sort(key=lambda item: float(item["cv_score"]), reverse=True)
        filtered = [item for item in prefilter_scores if float(item["cv_score"]) >= self.failure_probe_min_cv_score]
        if len(filtered) < self.failure_probe_force_keep:
            filtered = prefilter_scores[: self.failure_probe_force_keep]
        probe_indices = [int(item["index"]) for item in filtered[: self.failure_probe_top_k]]

        hits: list[dict[str, Any]] = []
        all_results: list[dict[str, Any]] = []
        raw_parts: list[str] = []
        t_start = time.perf_counter()
        for idx in probe_indices:
            before_idx = max(0, idx - 1)
            after_idx = min(len(sampled_frames) - 1, idx + 1)
            before_frame = sampled_frames[before_idx]
            center_frame = sampled_frames[idx]
            after_frame = sampled_frames[after_idx]
            result = self._run_failure_text_probe(
                before_image=before_frame.image_path,
                center_image=center_frame.image_path,
                after_image=after_frame.image_path,
                before_ts=before_frame.timestamp_sec,
                center_ts=center_frame.timestamp_sec,
                after_ts=after_frame.timestamp_sec,
                task_id=task_id,
                candidate_index=idx,
            )
            all_results.append(result)
            raw_parts.append(f"[failure_probe_idx_{idx:04d}]\n{result['raw_response']}")
            if result["load_failed"]:
                hits.append(result)

        best_hit = None
        if hits:
            best_hit = sorted(
                hits,
                key=lambda item: (
                    float(item["confidence"]) if item["confidence"] is not None else 0.0,
                    float(item["center_ts"]),
                ),
                reverse=True,
            )[0]
        return {
            "selected_indices": selected_indices,
            "probe_indices": probe_indices,
            "prefilter_scores": prefilter_scores,
            "results": all_results,
            "hits": hits,
            "best_hit": best_hit,
            "elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
            "raw_response": "\n\n".join(raw_parts),
        }

    def _run_loading_fallback(self, sampled_frames: list[Any], task_id: str) -> dict[str, Any]:
        """
        loading 兜底语义判断（首尾帧）。
        """
        before = sampled_frames[0]
        after = sampled_frames[-1]
        prompt_pack = build_prompt_for_type(
            task_type="loading",
            context={
                "task_type": "loading",
                "before_timestamp_sec": before.timestamp_sec,
                "after_timestamp_sec": after.timestamp_sec,
            },
        )
        t_vlm_start = time.perf_counter()
        eval_result = self.evaluator.evaluate_json(
            before_image=before.image_path,
            after_image=after.image_path,
            task_id=f"{task_id}_loading_fallback",
            system_prompt=prompt_pack.system_prompt,
            user_prompt=prompt_pack.user_prompt,
            required_fields={"bug_detected": bool, "reason": str, "decision_basis": str, "anomaly_type": str},
        )
        elapsed_ms = (time.perf_counter() - t_vlm_start) * 1000.0
        parsed = eval_result.parsed_json
        anomaly_type = str(parsed.get("anomaly_type", "unknown")).strip().lower() or "unknown"
        return {
            "bug_detected": bool(parsed["bug_detected"]),
            "reason": str(parsed["reason"]),
            "decision_basis": str(parsed.get("decision_basis", parsed["reason"])),
            "anomaly_type": anomaly_type,
            "raw_response": eval_result.raw_response,
            "elapsed_ms": round(elapsed_ms, 2),
        }

    def detect(self, sampled_frames: list[Any], task_id: str) -> LoadingDetectionResult:
        """对整段 sampled_frames 做 loading 检测。"""
        if len(sampled_frames) < 2:
            raise ValueError("loading 检测至少需要 2 帧")
        t_start = time.perf_counter()

        ratios: list[float] = []
        for idx in range(len(sampled_frames) - 1):
            gray_a = self._read_gray_image(sampled_frames[idx].image_path)
            gray_b = self._read_gray_image(sampled_frames[idx + 1].image_path)
            ratios.append(self._calc_change_ratio(gray_a, gray_b))

        screen_anomaly_type, screen_reason, screen_metrics = self._detect_black_white_screen(sampled_frames)
        cv_result, cv_reason, cv_anomaly_type, cv_metrics = self._cv_decide(ratios)
        cv_metrics["screen_stats"] = screen_metrics
        cv_state = "uncertain" if cv_result is None else ("positive" if cv_result else "negative")

        detected_anomaly_types: list[str] = []
        reason_parts: list[str] = []
        source_flags: list[str] = []
        raw_response_parts: list[str] = []

        if screen_anomaly_type is not None:
            detected_anomaly_types.append(screen_anomaly_type)
            reason_parts.append(screen_reason)
            source_flags.append("cv_black_white")

        probe_elapsed_ms = 0.0
        fallback_elapsed_ms = 0.0

        if cv_state == "positive":
            if cv_anomaly_type not in {"none", "unknown"}:
                detected_anomaly_types.append(cv_anomaly_type)
            reason_parts.append(cv_reason)
            source_flags.append("cv_loading_signal")
            # 性能优化：仅当 CV 已经识别到“长时间加载/无响应”时，才做失败提示语义探测。
            # 同时优先扫描末尾窗口，避免正常加载阶段的剧烈变化触发大量 VLM 调用。
            tail_window = min(len(sampled_frames), self.failure_probe_top_k)
            tail_indices = list(range(len(sampled_frames) - tail_window, len(sampled_frames)))
            probe_result = self._scan_failure_prompt_candidates(
                sampled_frames=sampled_frames,
                task_id=task_id,
                candidate_indices=tail_indices,
            )
            probe_elapsed_ms = float(probe_result["elapsed_ms"])
            best_probe_hit = probe_result["best_hit"]
            cv_metrics["failure_text_probe"] = {
                "enabled": True,
                "reason": "CV已识别到长时间加载/无响应，触发失败提示语义探测",
                "scan_selected_indices": probe_result["selected_indices"],
                "scan_probe_indices": probe_result["probe_indices"],
                "scan_count": len(probe_result["results"]),
                "scan_prefilter_scores": probe_result["prefilter_scores"],
                "hit_count": len(probe_result["hits"]),
                "load_failed": best_probe_hit is not None,
                "failure_type": (best_probe_hit["failure_type"] if best_probe_hit else "none"),
                "evidence_text": (best_probe_hit["evidence_text"] if best_probe_hit else ""),
                "confidence": (best_probe_hit["confidence"] if best_probe_hit else None),
                "reason_detail": (best_probe_hit["reason"] if best_probe_hit else "未命中明确失败语义"),
                "elapsed_ms": round(probe_elapsed_ms, 2),
            }
            if best_probe_hit is not None:
                detected_anomaly_types.append("load_failed")
                reason_parts.append(
                    f"失败探测命中：{best_probe_hit['reason']}（证据文案：{best_probe_hit['evidence_text']}，候选帧={best_probe_hit['candidate_index']}）"
                )
                source_flags.append("vlm_failure_probe")
            else:
                reason_parts.append("失败探测未命中明确失败语义。")
                source_flags.append("vlm_failure_probe")
            raw_response_parts.append(str(probe_result["raw_response"]))
        elif cv_state == "negative":
            reason_parts.append(cv_reason)
            source_flags.append("cv_clear_progress")
            cv_metrics["failure_text_probe"] = {
                "enabled": False,
                "reason": "CV未识别到长时间加载/无响应，跳过失败提示语义探测以降低VLM调用。",
                "scan_selected_indices": [],
                "scan_probe_indices": [],
                "scan_count": 0,
                "scan_prefilter_scores": [],
                "hit_count": 0,
                "load_failed": False,
                "failure_type": "none",
                "evidence_text": "",
                "confidence": None,
                "reason_detail": "",
                "elapsed_ms": 0.0,
            }
            reason_parts.append("失败探测已跳过（当前无长时间加载/无响应信号）。")
            source_flags.append("failure_probe_skipped")
        else:
            # cv_state == "uncertain"
            cv_metrics["failure_text_probe"] = {
                "enabled": False,
                "reason": "CV处于不确定态，先走loading语义兜底，不做失败提示探测。",
                "scan_selected_indices": [],
                "scan_probe_indices": [],
                "scan_count": 0,
                "scan_prefilter_scores": [],
                "hit_count": 0,
                "load_failed": False,
                "failure_type": "none",
                "evidence_text": "",
                "confidence": None,
                "reason_detail": "",
                "elapsed_ms": 0.0,
            }
            reason_parts.append(cv_reason)
            source_flags.append("cv_uncertain")
            fallback_result = self._run_loading_fallback(sampled_frames=sampled_frames, task_id=task_id)
            fallback_elapsed_ms = float(fallback_result["elapsed_ms"])
            reason_parts.append(f"VLM兜底：{fallback_result['reason']}")
            reason_parts.append(f"VLM依据：{fallback_result['decision_basis']}")
            source_flags.append("vlm_fallback")
            raw_response_parts.append(f"[loading_fallback]\n{fallback_result['raw_response']}")
            if bool(fallback_result["bug_detected"]) and str(fallback_result["anomaly_type"]) not in {"none"}:
                detected_anomaly_types.append(str(fallback_result["anomaly_type"]))

        normalized_anomaly_types = self._unique_keep_order(detected_anomaly_types)
        bug_detected = len([x for x in normalized_anomaly_types if x not in {"none"}]) > 0
        if not bug_detected:
            normalized_anomaly_types = ["none"]

        primary_anomaly = self._pick_primary_anomaly(normalized_anomaly_types)
        final_reason = " | ".join(reason_parts)
        decision_source = "multi_source" if len(set(source_flags)) > 1 else (source_flags[0] if source_flags else "unknown")
        task_intent = "检测页面是否处于异常长时间加载状态（疑似卡死或无反馈）。"

        return LoadingDetectionResult(
            bug_detected=bug_detected,
            task_intent=task_intent,
            reason=final_reason,
            decision_basis=final_reason,
            anomaly_type=primary_anomaly,
            anomaly_types=normalized_anomaly_types,
            raw_response="\n\n".join(raw_response_parts),
            decision_source=decision_source,
            cv_metrics=cv_metrics,
            timing={
                "detect_elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
                "failure_probe_elapsed_ms": round(probe_elapsed_ms, 2),
                "vlm_elapsed_ms": round(fallback_elapsed_ms, 2),
            },
        )
