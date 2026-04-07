"""GUI 前处理模块：为 VLM 提供结构化视觉证据。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2


@dataclass
class PreprocessEvidence:
    """一次前处理输出。"""

    summary: str
    structured: dict[str, Any]
    extra_image_paths: list[Path]


class GuiPreprocessor:
    """轻量 GUI 前处理器（ROI + 差分 + 候选区域）。"""

    def __init__(
        self,
        artifact_dir: Path | None = None,
        logger: logging.Logger | None = None,
        max_extra_images: int = 2,
        toast_min_contour_area: int = 1200,
        toast_size_target_ratio: float = 0.10,
        toast_size_tolerance: float = 0.10,
        toast_position_center_ratio: float = 0.50,
        toast_position_tolerance: float = 0.50,
        toast_weight_size: float = 0.55,
        toast_weight_position: float = 0.0,
        toast_weight_motion: float = 0.45,
        toast_motion_norm_ratio: float = 0.20,
        toast_dynamic_penalty_threshold: float = 0.45,
        toast_dynamic_penalty_scale: float = 1.2,
        toast_dynamic_penalty_max: float = 0.55,
        toast_candidate_max_area_ratio: float = 0.12,
        toast_candidate_max_height_ratio: float = 0.30,
        toast_candidate_min_aspect_ratio: float = 1.60,
        toast_candidate_expand_px: int = 10,
        toast_candidate_full_width_ratio: float = 0.96,
        toast_candidate_edge_touch_px: int = 3,
        toast_high_dynamic_threshold: float = 0.20,
        toast_band_search_ratio: float = 0.22,
        toast_band_min_area_scale: float = 0.40,
        toast_transition_penalty_threshold: float = 0.08,
        toast_transition_penalty_scale: float = 1.5,
        toast_transition_penalty_max: float = 0.45,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.logger = logger or logging.getLogger("vision_gui_agent")
        self.max_extra_images = max(0, int(max_extra_images))
        self.toast_min_contour_area = max(1, int(toast_min_contour_area))
        self.toast_size_target_ratio = max(0.0, float(toast_size_target_ratio))
        self.toast_size_tolerance = max(0.001, float(toast_size_tolerance))
        self.toast_position_center_ratio = min(1.0, max(0.0, float(toast_position_center_ratio)))
        self.toast_position_tolerance = max(0.001, float(toast_position_tolerance))
        self.toast_weight_size = max(0.0, float(toast_weight_size))
        self.toast_weight_position = max(0.0, float(toast_weight_position))
        self.toast_weight_motion = max(0.0, float(toast_weight_motion))
        self.toast_motion_norm_ratio = max(0.001, float(toast_motion_norm_ratio))
        self.toast_dynamic_penalty_threshold = min(1.0, max(0.0, float(toast_dynamic_penalty_threshold)))
        self.toast_dynamic_penalty_scale = max(0.0, float(toast_dynamic_penalty_scale))
        self.toast_dynamic_penalty_max = max(0.0, float(toast_dynamic_penalty_max))
        self.toast_candidate_max_area_ratio = min(1.0, max(0.001, float(toast_candidate_max_area_ratio)))
        self.toast_candidate_max_height_ratio = min(1.0, max(0.001, float(toast_candidate_max_height_ratio)))
        self.toast_candidate_min_aspect_ratio = max(0.1, float(toast_candidate_min_aspect_ratio))
        self.toast_candidate_expand_px = max(0, int(toast_candidate_expand_px))
        self.toast_candidate_full_width_ratio = min(1.0, max(0.5, float(toast_candidate_full_width_ratio)))
        self.toast_candidate_edge_touch_px = max(0, int(toast_candidate_edge_touch_px))
        self.toast_high_dynamic_threshold = min(1.0, max(0.0, float(toast_high_dynamic_threshold)))
        self.toast_band_search_ratio = min(1.0, max(0.05, float(toast_band_search_ratio)))
        self.toast_band_min_area_scale = min(1.0, max(0.05, float(toast_band_min_area_scale)))
        self.toast_transition_penalty_threshold = min(1.0, max(0.0, float(toast_transition_penalty_threshold)))
        self.toast_transition_penalty_scale = max(0.0, float(toast_transition_penalty_scale))
        self.toast_transition_penalty_max = max(0.0, float(toast_transition_penalty_max))

    @staticmethod
    def _normalized_distance_score(value: float, center: float, tolerance: float) -> float:
        return max(0.0, 1.0 - abs(value - center) / max(1e-6, tolerance))

    @staticmethod
    def _shape_score(aspect_ratio: float) -> float:
        # toast 往往是扁平横条，宽高比过小通常是误检区域。
        return max(0.0, min(1.0, (aspect_ratio - 1.0) / 2.0))

    def _collect_toast_candidates(
        self,
        binary: Any,
        width: int,
        height: int,
        min_contour_area: int | None = None,
        y_offset: int = 0,
        source: str = "diff_contour",
    ) -> list[dict[str, Any]]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(max(1, width * height))
        area_threshold = self.toast_min_contour_area if min_contour_area is None else max(1, int(min_contour_area))
        candidates: list[dict[str, Any]] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = float(bw * bh)
            if area < area_threshold:
                continue
            area_ratio = area / frame_area
            width_ratio = float(bw / max(1, width))
            height_ratio = float(bh / max(1, height))
            aspect_ratio = float(bw / max(1, bh))
            if area_ratio > self.toast_candidate_max_area_ratio:
                continue
            if height_ratio > self.toast_candidate_max_height_ratio:
                continue
            if aspect_ratio < self.toast_candidate_min_aspect_ratio:
                continue
            left_touch = x <= self.toast_candidate_edge_touch_px
            right_touch = (x + bw) >= (width - self.toast_candidate_edge_touch_px)
            if left_touch and right_touch and width_ratio >= self.toast_candidate_full_width_ratio:
                # 贴两侧边缘的近全宽条带通常是页面固定栏，不是浮层 toast。
                continue

            abs_y = y + max(0, int(y_offset))
            center_y_ratio = (abs_y + (bh / 2.0)) / max(1.0, float(height))
            size_score = self._normalized_distance_score(
                value=area_ratio,
                center=self.toast_size_target_ratio,
                tolerance=self.toast_size_tolerance,
            )
            position_score = self._normalized_distance_score(
                value=center_y_ratio,
                center=self.toast_position_center_ratio,
                tolerance=self.toast_position_tolerance,
            )
            shape_score = self._shape_score(aspect_ratio=aspect_ratio)
            candidate_score = 0.45 * size_score + 0.35 * position_score + 0.20 * shape_score

            ex = max(0, x - self.toast_candidate_expand_px)
            ey = max(0, abs_y - self.toast_candidate_expand_px)
            ex2 = min(width, x + bw + self.toast_candidate_expand_px)
            ey2 = min(height, abs_y + bh + self.toast_candidate_expand_px)
            candidates.append(
                {
                    "box": (ex, ey, max(1, ex2 - ex), max(1, ey2 - ey)),
                    "raw_box": (x, abs_y, bw, bh),
                    "area_ratio": area_ratio,
                    "width_ratio": width_ratio,
                    "height_ratio": height_ratio,
                    "aspect_ratio": aspect_ratio,
                    "center_y_ratio": center_y_ratio,
                    "size_score": size_score,
                    "position_score": position_score,
                    "shape_score": shape_score,
                    "candidate_score": candidate_score,
                    "source": source,
                }
            )

        candidates.sort(key=lambda item: float(item["candidate_score"]), reverse=True)
        return candidates

    def _collect_toast_candidates_with_fallback(
        self,
        binary: Any,
        width: int,
        height: int,
        global_change_ratio: float,
    ) -> list[dict[str, Any]]:
        candidates = self._collect_toast_candidates(binary=binary, width=width, height=height)
        if candidates or global_change_ratio < self.toast_high_dynamic_threshold:
            return candidates

        band_h = max(24, int(height * self.toast_band_search_ratio))
        band_min_area = max(1, int(self.toast_min_contour_area * self.toast_band_min_area_scale))
        top_binary = binary[0:band_h, :]
        bottom_y = max(0, height - band_h)
        bottom_binary = binary[bottom_y:height, :]
        top_candidates = self._collect_toast_candidates(
            binary=top_binary,
            width=width,
            height=height,
            min_contour_area=band_min_area,
            y_offset=0,
            source="high_dynamic_top_band",
        )
        bottom_candidates = self._collect_toast_candidates(
            binary=bottom_binary,
            width=width,
            height=height,
            min_contour_area=band_min_area,
            y_offset=bottom_y,
            source="high_dynamic_bottom_band",
        )
        merged = top_candidates + bottom_candidates
        merged.sort(key=lambda item: float(item["candidate_score"]), reverse=True)
        return merged

    def _build_toast_diff_binary(self, before: Any, center: Any, after: Any) -> Any:
        before_diff = cv2.absdiff(center, before)
        after_diff = cv2.absdiff(center, after)
        before_gray = cv2.cvtColor(before_diff, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_diff, cv2.COLOR_BGR2GRAY)
        merged = cv2.max(before_gray, after_gray)
        blurred = cv2.GaussianBlur(merged, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 22, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    @staticmethod
    def _read_image(path: Path) -> Any:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {path}")
        return image

    @staticmethod
    def _clamp_bounds(width: int, height: int, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
        x1 = max(0, min(int(x), width - 1))
        y1 = max(0, min(int(y), height - 1))
        x2 = max(x1 + 1, min(int(x) + int(w), width))
        y2 = max(y1 + 1, min(int(y) + int(h), height))
        return x1, y1, x2, y2

    def _save_artifact(self, image: Any, file_name: str) -> Path | None:
        if self.artifact_dir is None:
            return None
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / file_name
        ok = cv2.imwrite(str(path), image)
        if not ok:
            self.logger.warning("前处理图保存失败: %s", path)
            return None
        return path

    @staticmethod
    def _calc_diff_metrics(before_roi: Any, after_roi: Any) -> tuple[float, float, Any]:
        diff = cv2.absdiff(before_roi, after_roi)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(diff_gray, 20, 255, cv2.THRESH_BINARY)
        changed_pixels = int(cv2.countNonZero(binary))
        total_pixels = max(1, int(binary.shape[0] * binary.shape[1]))
        change_ratio = changed_pixels / total_pixels
        mean_abs_diff = float(diff_gray.mean())
        return mean_abs_diff, change_ratio, diff_gray

    @staticmethod
    def _calc_frame_change_ratio(image_a: Any, image_b: Any, threshold: int = 22) -> float:
        diff = cv2.absdiff(image_a, image_b)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
        changed_pixels = int(cv2.countNonZero(binary))
        total_pixels = max(1, int(binary.shape[0] * binary.shape[1]))
        return changed_pixels / total_pixels

    def prepare_count_change(
        self,
        before_image: Path,
        after_image: Path,
        bounds: dict[str, int],
        task_id: str,
    ) -> PreprocessEvidence:
        """构造 count_change 场景证据。"""
        t_start = time.perf_counter()
        before = self._read_image(before_image)
        after = self._read_image(after_image)
        h, w = before.shape[:2]
        x1, y1, x2, y2 = self._clamp_bounds(
            width=w,
            height=h,
            x=int(bounds["x"]),
            y=int(bounds["y"]),
            w=int(bounds["width"]),
            h=int(bounds["height"]),
        )

        before_roi = before[y1:y2, x1:x2]
        after_roi = after[y1:y2, x1:x2]
        mean_abs_diff, change_ratio, diff_gray = self._calc_diff_metrics(before_roi, after_roi)

        out_paths: list[Path] = []
        before_path = self._save_artifact(before_roi, f"{task_id}_pp_count_before_roi.png")
        after_path = self._save_artifact(after_roi, f"{task_id}_pp_count_after_roi.png")
        diff_path = self._save_artifact(diff_gray, f"{task_id}_pp_count_diff_gray.png")
        for path in [before_path, after_path, diff_path]:
            if path is not None:
                out_paths.append(path)

        structured = {
            "evidence_type": "count_change_roi_diff",
            "roi_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "roi_mean_abs_diff": round(mean_abs_diff, 3),
            "roi_changed_pixel_ratio": round(change_ratio, 4),
            "notes": "该指标描述控件区域前后变化强度，仅用于辅助 VLM 注意力聚焦。",
        }
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        structured["preprocess_elapsed_ms"] = round(elapsed_ms, 2)
        summary = (
            "前处理证据(count_change): "
            f"控件ROI=({x1},{y1})-({x2},{y2}), "
            f"mean_abs_diff={mean_abs_diff:.3f}, changed_ratio={change_ratio:.4f}, elapsed={elapsed_ms:.2f}ms"
        )
        return PreprocessEvidence(
            summary=summary,
            structured=structured,
            extra_image_paths=out_paths[: self.max_extra_images],
        )

    def prepare_toast_triplet(
        self,
        before_image: Path,
        center_image: Path,
        after_image: Path,
        task_id: str,
        candidate_index: int,
    ) -> PreprocessEvidence:
        """构造 toast 候选帧证据（差分候选框 + ROI 裁剪）。"""
        t_start = time.perf_counter()
        before = self._read_image(before_image)
        center = self._read_image(center_image)
        after = self._read_image(after_image)
        h, w = center.shape[:2]
        binary = self._build_toast_diff_binary(before=before, center=center, after=after)
        changed_pixels = int(cv2.countNonZero(binary))
        total_pixels = max(1, int(h * w))
        global_change_ratio = changed_pixels / total_pixels
        before_after_change_ratio = self._calc_frame_change_ratio(before, after)
        candidates = self._collect_toast_candidates_with_fallback(
            binary=binary,
            width=w,
            height=h,
            global_change_ratio=global_change_ratio,
        )
        best_box: tuple[int, int, int, int] | None = candidates[0]["box"] if candidates else None

        source = str(candidates[0]["source"]) if candidates else "diff_contour"
        if best_box is None:
            source = "fallback_bottom_band"
            fallback_h = max(40, int(h * 0.20))
            best_box = (0, h - fallback_h, w, fallback_h)

        bx, by, bw, bh = best_box
        x1, y1, x2, y2 = self._clamp_bounds(width=w, height=h, x=bx, y=by, w=bw, h=bh)
        center_roi = center[y1:y2, x1:x2]
        after_roi = after[y1:y2, x1:x2]
        mean_abs_diff, change_ratio, _ = self._calc_diff_metrics(center_roi, after_roi)

        suffix = f"{task_id}_pp_toast_{candidate_index:04d}"
        out_paths: list[Path] = []
        roi_center_path = self._save_artifact(center_roi, f"{suffix}_center_roi.png")
        roi_after_path = self._save_artifact(after_roi, f"{suffix}_after_roi.png")
        binary_path = self._save_artifact(binary, f"{suffix}_diff_mask.png")
        for path in [roi_center_path, roi_after_path, binary_path]:
            if path is not None:
                out_paths.append(path)

        structured = {
            "evidence_type": "toast_candidate_roi_diff",
            "candidate_index": candidate_index,
            "candidate_roi_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "candidate_source": source,
            "candidate_count": len(candidates),
            "candidate_top_scores": [round(float(item["candidate_score"]), 4) for item in candidates[:3]],
            "global_change_ratio": round(global_change_ratio, 4),
            "roi_mean_abs_diff_center_to_after": round(mean_abs_diff, 3),
            "roi_changed_pixel_ratio_center_to_after": round(change_ratio, 4),
            "notes": "该 ROI 为差分候选区域，可能覆盖 toast 或局部动态区域。",
        }
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        structured["preprocess_elapsed_ms"] = round(elapsed_ms, 2)
        summary = (
            "前处理证据(toast): "
            f"candidate_idx={candidate_index}, source={source}, "
            f"roi=({x1},{y1})-({x2},{y2}), diff_mean={mean_abs_diff:.3f}, changed_ratio={change_ratio:.4f}, elapsed={elapsed_ms:.2f}ms"
        )
        return PreprocessEvidence(
            summary=summary,
            structured=structured,
            extra_image_paths=out_paths[: self.max_extra_images],
        )

    def score_toast_triplet(
        self,
        before_image: Path,
        center_image: Path,
        after_image: Path,
    ) -> dict[str, Any]:
        """
        对 toast 候选三帧做快速打分，用于关键帧筛选（不调用 VLM）。
        """
        t_start = time.perf_counter()
        before = self._read_image(before_image)
        center = self._read_image(center_image)
        after = self._read_image(after_image)
        h, w = center.shape[:2]
        binary = self._build_toast_diff_binary(before=before, center=center, after=after)

        changed_pixels = int(cv2.countNonZero(binary))
        total_pixels = max(1, int(h * w))
        global_change_ratio = changed_pixels / total_pixels
        before_after_change_ratio = self._calc_frame_change_ratio(before, after)

        candidates = self._collect_toast_candidates_with_fallback(
            binary=binary,
            width=w,
            height=h,
            global_change_ratio=global_change_ratio,
        )
        best_box: tuple[int, int, int, int] | None = candidates[0]["box"] if candidates else None

        if best_box is None:
            return {
                "score": 0.0,
                "global_change_ratio": round(global_change_ratio, 4),
                "candidate_roi_box": None,
                "source": "no_contour",
                "reason": "未找到满足几何/面积约束的候选区域（高动态兜底检索后仍为空）",
            }

        bx, by, bw, bh = best_box
        top = candidates[0]
        area_ratio = float(top["area_ratio"])
        center_y_ratio = float(top["center_y_ratio"])
        size_score = float(top["size_score"])
        position_score = float(top["position_score"])
        shape_score = float(top["shape_score"])
        geom_score = float(top["candidate_score"])
        motion_score = min(1.0, global_change_ratio / self.toast_motion_norm_ratio)
        total_weight = self.toast_weight_size + self.toast_weight_position + self.toast_weight_motion
        if total_weight <= 1e-6:
            # 避免全 0 权重导致无法排序，退化为 motion_score。
            total_weight = 1.0
            weighted_base = motion_score
        else:
            weighted_base = (
                self.toast_weight_size * geom_score
                + self.toast_weight_position * position_score
                + self.toast_weight_motion * motion_score
            ) / total_weight
        dynamic_penalty = 0.0
        if global_change_ratio > self.toast_dynamic_penalty_threshold:
            dynamic_penalty = min(
                self.toast_dynamic_penalty_max,
                (global_change_ratio - self.toast_dynamic_penalty_threshold) * self.toast_dynamic_penalty_scale,
            )
        source = str(top.get("source", "diff_contour"))
        transition_penalty = 0.0
        if source == "diff_contour" and before_after_change_ratio > self.toast_transition_penalty_threshold:
            transition_penalty = min(
                self.toast_transition_penalty_max,
                (before_after_change_ratio - self.toast_transition_penalty_threshold) * self.toast_transition_penalty_scale,
            )
        final_score = max(0.0, weighted_base - dynamic_penalty - transition_penalty)

        x1, y1, x2, y2 = self._clamp_bounds(width=w, height=h, x=bx, y=by, w=bw, h=bh)
        return {
            "score": round(final_score, 4),
            "score_elapsed_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
            "global_change_ratio": round(global_change_ratio, 4),
            "before_after_change_ratio": round(before_after_change_ratio, 4),
            "area_ratio": round(area_ratio, 4),
            "center_y_ratio": round(center_y_ratio, 4),
            "score_components": {
                "size_score": round(size_score, 4),
                "position_score": round(position_score, 4),
                "shape_score": round(shape_score, 4),
                "geom_score": round(geom_score, 4),
                "motion_score": round(motion_score, 4),
                "weighted_base": round(weighted_base, 4),
                "dynamic_penalty": round(dynamic_penalty, 4),
                "transition_penalty": round(transition_penalty, 4),
            },
            "weights": {
                "size": round(self.toast_weight_size, 4),
                "position": round(self.toast_weight_position, 4),
                "motion": round(self.toast_weight_motion, 4),
            },
            "candidate_roi_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "candidate_count": len(candidates),
            "candidate_aspect_ratio": round(float(top["aspect_ratio"]), 4),
            "candidate_height_ratio": round(float(top["height_ratio"]), 4),
            "source": source,
            "reason": "基于几何约束+动态强度+转场惩罚的加权评分，可通过环境变量调参",
        }
