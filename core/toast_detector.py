"""Toast 消息检测模块：定位关键帧并校验文案是否符合预期。"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.evaluator import VisionEvaluator
from core.perception import ExtractedFrame
from core.preprocessor import GuiPreprocessor
from core.prompt_builders import build_prompt_for_type, render_toast_user_prompt


@dataclass
class ToastDetectionResult:
    """Toast 检测结果。"""

    bug_detected: bool
    expectation_met: bool
    task_intent: str
    key_frame_path: Path | None
    key_frame_timestamp: float | None
    toast_text: str
    action_semantic: str
    inferred_expected_toast_text: str
    reason: str
    confidence: float | None
    raw_response: str
    scanned_candidates: int
    total_candidates: int
    evaluated_candidate_indices: list[int]
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] | None = None
    preprocess_evidence: dict[str, Any] | None = None


class ToastMessageDetector:
    """基于 VLM 的 Toast 关键帧定位与语义判定。"""

    def __init__(
        self,
        evaluator: VisionEvaluator,
        logger: logging.Logger | None = None,
        debug: bool = False,
        preprocessor: GuiPreprocessor | None = None,
        enable_preprocess: bool = True,
        top_k_candidates: int = 3,
        prompt_version: str = "current",
        # early_stop_confidence: float = 0.95,
    ) -> None:
        self.evaluator = evaluator
        self.logger = logger or logging.getLogger("vision_gui_agent")
        self.debug = debug
        self.preprocessor = preprocessor
        self.enable_preprocess = enable_preprocess
        self.top_k_candidates = max(1, int(top_k_candidates))
        self.prompt_pack = build_prompt_for_type(
            task_type="toast",
            context={
                "prompt_version": prompt_version,
                "logger": self.logger,
            },
        )
        # self.early_stop_confidence = min(1.0, max(0.0, float(early_stop_confidence)))

    @staticmethod
    def _normalize_toast_text(text: str) -> str:
        normalized = (text or "").strip().lower()
        normalized = re.sub(r"[，。！？、,:;；!?\s]+", "", normalized)
        # 常见业务同义词归一，避免“房间/直播间”类文案差异造成误判。
        normalized = normalized.replace("直播间", "房间")
        # 弱语义后缀，通常不影响主干语义。
        for noise in ["请稍后重试", "稍后重试", "请重试", "请稍候再试", "请稍后再试"]:
            normalized = normalized.replace(noise, "")
        return normalized

    @staticmethod
    def _has_failure_signal(text: str) -> bool:
        return any(token in text for token in ["失败", "错误", "异常", "超时", "无法", "不能"])

    @staticmethod
    def _has_success_signal(text: str) -> bool:
        return any(token in text for token in ["成功", "已", "完成", "恢复", "还原", "通过"])

    @staticmethod
    def _is_likely_cta_text(text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        cta_tokens = ["点击", "进入", "去", "立即", "马上", "查看", "开启", "直播间"]
        return any(token in normalized for token in cta_tokens)

    @staticmethod
    def _char_ngram_set(text: str, n: int) -> set[str]:
        if len(text) < n:
            return {text} if text else set()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    @classmethod
    def _semantic_equivalent_toast(cls, actual_text: str, expected_text: str) -> bool:
        actual_norm = cls._normalize_toast_text(actual_text)
        expected_norm = cls._normalize_toast_text(expected_text)
        if not actual_norm or not expected_norm:
            return False
        if actual_norm == expected_norm:
            return True
        if actual_norm in expected_norm or expected_norm in actual_norm:
            return True

        actual_fail = cls._has_failure_signal(actual_norm)
        expected_fail = cls._has_failure_signal(expected_norm)
        actual_success = cls._has_success_signal(actual_norm)
        expected_success = cls._has_success_signal(expected_norm)
        # 明显成败冲突时，不能认定为语义一致。
        if (actual_fail and expected_success and not expected_fail) or (
            expected_fail and actual_success and not actual_fail
        ):
            return False

        actual_2gram, expected_2gram = cls._char_ngram_set(actual_norm, 2), cls._char_ngram_set(expected_norm, 2)
        actual_3gram, expected_3gram = cls._char_ngram_set(actual_norm, 3), cls._char_ngram_set(expected_norm, 3)
        jaccard_2 = (
            (len(actual_2gram & expected_2gram) / len(actual_2gram | expected_2gram))
            if (actual_2gram and expected_2gram)
            else 0.0
        )
        jaccard_3 = (
            (len(actual_3gram & expected_3gram) / len(actual_3gram | expected_3gram))
            if (actual_3gram and expected_3gram)
            else 0.0
        )
        return max(jaccard_2, jaccard_3) >= 0.33

    @staticmethod
    def _build_toast_image_role_labels(extra_image_paths: list[Path]) -> list[str]:
        labels = [
            "图1:动作前完整帧（必须用于推断操作语义）",
            "图2:候选完整帧（可能包含toast）",
        ]
        for idx, path in enumerate(extra_image_paths):
            name = path.name.lower()
            if idx == 0:
                labels.append("图3:后续完整帧（用于判断是否为瞬时提示）")
                continue
            if "after_roi" in name:
                labels.append("局部ROI:候选帧toast区域裁剪")
            elif "center_roi" in name:
                labels.append("局部ROI:后续帧同区域裁剪")
            elif "diff_mask" in name:
                labels.append("辅助图:差分mask（仅用于定位，不代表语义）")
            else:
                labels.append("额外证据图（后续完整帧）")
        return labels

    @staticmethod
    def _compact_preprocess_structured(preprocess_structured: dict[str, Any] | None) -> dict[str, Any]:
        """压缩前处理证据，减少 prompt 噪声与 token 开销。"""
        if not preprocess_structured:
            return {}
        return {
            "candidate_index": preprocess_structured.get("candidate_index"),
            "candidate_source": preprocess_structured.get("candidate_source"),
            "candidate_roi_box": preprocess_structured.get("candidate_roi_box"),
            "roi_changed_ratio": preprocess_structured.get("roi_changed_pixel_ratio_center_to_after"),
            "roi_mean_abs_diff": preprocess_structured.get("roi_mean_abs_diff_center_to_after"),
        }

    @staticmethod
    def _select_final_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None

        def _rank_key(candidate: dict[str, Any]) -> tuple[float, int]:
            return (float(candidate.get("confidence") or 0.0), int(candidate.get("idx") or 0))

        visible_reliable_failures = [
            c
            for c in candidates
            if bool(c.get("toast_visible"))
            and (not bool(c.get("expectation_met")))
            and (not bool(c.get("is_uncertain_action")))
        ]
        if visible_reliable_failures:
            # 仅“可解释且低风险”的冲突候选优先，避免不确定动作导致误报。
            return max(visible_reliable_failures, key=_rank_key)

        visible_expectations_met = [c for c in candidates if bool(c.get("toast_visible")) and bool(c.get("expectation_met"))]
        if visible_expectations_met:
            return max(visible_expectations_met, key=_rank_key)

        visible_uncertain_failures = [
            c
            for c in candidates
            if bool(c.get("toast_visible"))
            and (not bool(c.get("expectation_met")))
            and bool(c.get("is_uncertain_action"))
        ]
        if visible_uncertain_failures:
            return max(visible_uncertain_failures, key=_rank_key)

        return max(candidates, key=_rank_key)

    def detect(
        self,
        sampled_frames: list[ExtractedFrame],
        task_id: str,
        expected_toast_keywords: list[str] | None = None,
    ) -> ToastDetectionResult:
        """
        自动扫描帧序列中的 Toast，并判断文案是否符合预期。
        """
        if len(sampled_frames) < 2:
            raise ValueError("toast 检测至少需要 2 帧")

        task_intent = self.prompt_pack.task_intent or "检测动作触发后的 toast 文案是否与预期语义一致。"
        keywords_text = "、".join(expected_toast_keywords or []) if expected_toast_keywords else "无"

        evaluated_candidates: list[dict[str, Any]] = []
        scanned = 0
        total_candidates = len(sampled_frames)
        detect_start = time.perf_counter()
        scoring_elapsed_ms = 0.0
        preprocess_elapsed_total_ms = 0.0
        eval_elapsed_total_ms = 0.0
        candidate_scores: list[dict[str, Any]] = []

        candidate_indices = list(range(total_candidates))
        if self.enable_preprocess and self.preprocessor is not None and total_candidates > self.top_k_candidates:
            t_scoring_start = time.perf_counter()
            scored: list[tuple[int, float]] = []
            for idx in candidate_indices:
                center = sampled_frames[idx]
                before = sampled_frames[idx - 1] if idx > 0 else sampled_frames[idx]
                after = sampled_frames[idx + 1] if idx < total_candidates - 1 else sampled_frames[idx]
                try:
                    score_result = self.preprocessor.score_toast_triplet(
                        before_image=before.image_path,
                        center_image=center.image_path,
                        after_image=after.image_path,
                    )
                    score_value = float(score_result.get("score", 0.0))
                    scored.append((idx, score_value))
                    candidate_scores.append(
                        {
                            "index": idx,
                            "timestamp_sec": center.timestamp_sec,
                            "score": score_value,
                            **score_result,
                        }
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    if self.debug:
                        self.logger.warning("toast 候选打分失败: idx=%s, err=%s", idx, exc)
                    scored.append((idx, 0.0))
                    candidate_scores.append(
                        {
                            "index": idx,
                            "timestamp_sec": center.timestamp_sec,
                            "score": 0.0,
                            "source": "score_error",
                            "reason": str(exc),
                        }
                    )

            scored.sort(key=lambda x: x[1], reverse=True)
            candidate_scores.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            candidate_indices = [idx for idx, _ in scored[: self.top_k_candidates]]
            candidate_indices.sort()
            scoring_elapsed_ms = (time.perf_counter() - t_scoring_start) * 1000.0
            if self.debug:
                self.logger.info(
                    "toast 候选筛选: total=%s, top_k=%s, selected=%s, elapsed=%.2fms",
                    total_candidates,
                    self.top_k_candidates,
                    candidate_indices,
                    scoring_elapsed_ms,
                )

        for idx in candidate_indices:
            center = sampled_frames[idx]
            before = sampled_frames[idx - 1] if idx > 0 else sampled_frames[idx]
            after = sampled_frames[idx + 1] if idx < len(sampled_frames) - 1 else sampled_frames[idx]
            segment_task_id = f"{task_id}_toast_scan_{idx:04d}"
            preprocess_summary = "无"
            preprocess_structured: dict[str, Any] | None = None
            preprocess_extra_images: list[Path] = []
            if self.enable_preprocess and self.preprocessor is not None:
                try:
                    t_preprocess_start = time.perf_counter()
                    evidence = self.preprocessor.prepare_toast_triplet(
                        before_image=before.image_path,
                        center_image=center.image_path,
                        after_image=after.image_path,
                        task_id=task_id,
                        candidate_index=idx,
                    )
                    preprocess_summary = evidence.summary
                    preprocess_structured = evidence.structured
                    preprocess_extra_images = evidence.extra_image_paths
                    preprocess_elapsed_total_ms += (time.perf_counter() - t_preprocess_start) * 1000.0
                except Exception as exc:  # pylint: disable=broad-except
                    if self.debug:
                        self.logger.warning("toast 前处理失败: idx=%s, err=%s", idx, exc)

            system_prompt = self.prompt_pack.system_prompt
            user_prompt = render_toast_user_prompt(
                prompt_pack=self.prompt_pack,
                context={
                    "task_intent": task_intent,
                    "candidate_timestamp_sec": center.timestamp_sec,
                    "keywords_text": keywords_text,
                    "preprocess_summary": preprocess_summary,
                    "preprocess_structured_json": json.dumps(
                        self._compact_preprocess_structured(preprocess_structured),
                        ensure_ascii=False,
                    )
                    if preprocess_structured
                    else "{}",
                },
            )

            scanned += 1
            try:
                t_eval_start = time.perf_counter()
                result = self.evaluator.evaluate_json(
                    before_image=before.image_path,
                    after_image=center.image_path,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    task_id=segment_task_id,
                    required_fields={
                        "toast_visible": bool,
                        "toast_text": str,
                        "action_semantic": str,
                        "inferred_expected_toast_text": str,
                        "expectation_met": (bool, type(None)),
                        "reverse_inference_risk": str,
                        "action_evidence_from_frame12": str,
                        "toast_evidence_from_frame23": str,
                        "reason": str,
                    },
                    # extra_image_paths=[after.image_path] + preprocess_extra_images,
                    extra_image_paths=[after.image_path],
                    # image_role_labels=self._build_toast_image_role_labels([after.image_path] + preprocess_extra_images),
                    image_role_labels=self._build_toast_image_role_labels([after.image_path]),
                )
                eval_elapsed_total_ms += (time.perf_counter() - t_eval_start) * 1000.0
            except Exception as exc:  # pylint: disable=broad-except
                if self.debug:
                    self.logger.warning("toast 扫描失败: idx=%s, err=%s", idx, exc)
                continue

            parsed = result.parsed_json
            confidence = float(parsed["confidence"]) if parsed.get("confidence") is not None else 0.0
            toast_text = str(parsed.get("toast_text", ""))
            action_semantic = str(parsed.get("action_semantic", ""))
            inferred_expected_toast_text = str(parsed.get("inferred_expected_toast_text", ""))
            raw_expectation = parsed.get("expectation_met", False)
            raw_expectation_met = bool(raw_expectation) if isinstance(raw_expectation, bool) else False
            toast_visible = bool(parsed["toast_visible"])
            reverse_inference_risk = str(parsed.get("reverse_inference_risk", "low")).strip().lower()
            action_evidence_from_frame12 = str(parsed.get("action_evidence_from_frame12", "")).strip()
            toast_evidence_from_frame23 = str(parsed.get("toast_evidence_from_frame23", "")).strip()
            action_semantic_norm = action_semantic.strip().lower()
            expectation_unknown = raw_expectation is None
            if action_semantic_norm in {"unknown", "uncertain", "不确定", "无法确定", "未知"}:
                expectation_unknown = True

            # 硬门控：未检测到 toast 时，不继续做文案/预期判断，降低结果抖动。
            if not toast_visible:
                toast_text = ""
                inferred_expected_toast_text = ""
                raw_expectation_met = False

            semantic_equivalent = self._semantic_equivalent_toast(toast_text, inferred_expected_toast_text)
            expectation_met = raw_expectation_met or semantic_equivalent
            reason = str(parsed.get("reason", ""))
            if reverse_inference_risk not in {"low", "high"}:
                reverse_inference_risk = "high"
            is_uncertain_action = expectation_unknown or (reverse_inference_risk == "high")
            if expectation_unknown:
                expectation_met = False

            roi_changed_ratio = None
            if preprocess_structured is not None:
                try:
                    roi_changed_ratio = float(preprocess_structured.get("roi_changed_pixel_ratio_center_to_after"))
                except (TypeError, ValueError):
                    roi_changed_ratio = None
            text_norm = self._normalize_toast_text(toast_text)
            is_cta_like = self._is_likely_cta_text(toast_text)
            has_status_signal = self._has_failure_signal(text_norm) or self._has_success_signal(text_norm)
            if toast_visible and roi_changed_ratio is not None and roi_changed_ratio < 0.015 and is_cta_like and not has_status_signal:
                toast_visible = False
                expectation_met = False
                reason = (
                    f"{reason}（后处理修正：识别文本更像页面CTA且ROI在后续帧几乎不变"
                    f"(changed_ratio={roi_changed_ratio:.4f})，按非toast处理。）"
                ).strip()

            if expectation_met and (not raw_expectation_met) and semantic_equivalent:
                reason = f"{reason}（后处理修正：文案非逐字一致，但语义一致，按 expectation_met=true 处理。）".strip()
            if toast_visible and expectation_met and reverse_inference_risk == "high":
                expectation_met = False
                reason = (
                    f"{reason}（后处理修正：模型标记存在反向推断风险，按 expectation_met=false 保守处理。）"
                ).strip()
            if toast_visible and (not expectation_met) and is_uncertain_action:
                reason = f"{reason}（后处理修正：动作语义可观测性不足，按不确定处理。）".strip()
            candidate = {
                "idx": idx,
                "frame": center,
                "toast_visible": toast_visible,
                "toast_text": toast_text,
                "action_semantic": action_semantic,
                "inferred_expected_toast_text": inferred_expected_toast_text,
                "expectation_met": expectation_met,
                "is_uncertain_action": is_uncertain_action,
                "reason": reason,
                "confidence": confidence,
                "raw_response": result.raw_response,
                "preprocess_evidence": {
                    **(preprocess_structured or {}),
                    "reverse_inference_risk": reverse_inference_risk,
                    "action_evidence_from_frame12": action_evidence_from_frame12,
                    "toast_evidence_from_frame23": toast_evidence_from_frame23,
                },
            }

            evaluated_candidates.append(candidate)

        best_candidate = self._select_final_candidate(evaluated_candidates)

        if best_candidate is None:
            detect_elapsed_ms = (time.perf_counter() - detect_start) * 1000.0
            return ToastDetectionResult(
                bug_detected=False,
                expectation_met=False,
                task_intent=task_intent,
                key_frame_path=None,
                key_frame_timestamp=None,
                toast_text="",
                action_semantic="",
                inferred_expected_toast_text="",
                reason="未能从候选帧中识别到有效 toast 结果。",
                confidence=None,
                raw_response="",
                scanned_candidates=scanned,
                total_candidates=total_candidates,
                evaluated_candidate_indices=candidate_indices,
                candidate_scores=candidate_scores,
                timing={
                    "detect_elapsed_ms": round(detect_elapsed_ms, 2),
                    "scoring_elapsed_ms": round(scoring_elapsed_ms, 2),
                    "preprocess_elapsed_total_ms": round(preprocess_elapsed_total_ms, 2),
                    "vlm_eval_elapsed_total_ms": round(eval_elapsed_total_ms, 2),
                },
                preprocess_evidence=None,
            )

        if best_candidate["toast_visible"]:
            bug_detected = (not best_candidate["expectation_met"]) and (not bool(best_candidate.get("is_uncertain_action")))
        else:
            bug_detected = False
        detect_elapsed_ms = (time.perf_counter() - detect_start) * 1000.0
        return ToastDetectionResult(
            bug_detected=bug_detected,
            expectation_met=best_candidate["expectation_met"],
            task_intent=task_intent,
            key_frame_path=best_candidate["frame"].image_path,
            key_frame_timestamp=best_candidate["frame"].timestamp_sec,
            toast_text=best_candidate["toast_text"],
            action_semantic=best_candidate["action_semantic"],
            inferred_expected_toast_text=best_candidate["inferred_expected_toast_text"],
            reason=best_candidate["reason"],
            confidence=best_candidate["confidence"],
            raw_response=best_candidate["raw_response"],
            scanned_candidates=scanned,
            total_candidates=total_candidates,
            evaluated_candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            timing={
                "detect_elapsed_ms": round(detect_elapsed_ms, 2),
                "scoring_elapsed_ms": round(scoring_elapsed_ms, 2),
                "preprocess_elapsed_total_ms": round(preprocess_elapsed_total_ms, 2),
                "vlm_eval_elapsed_total_ms": round(eval_elapsed_total_ms, 2),
            },
            preprocess_evidence=best_candidate.get("preprocess_evidence"),
        )
