"""内容列表刷新检测模块：判定控件触发后目标区域是否刷新。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2

from core.count_change_detector import ControlBounds
from core.evaluator import VisionEvaluator
from core.prompt_builders import build_prompt_for_type


@dataclass
class ListRefreshResult:
    """列表刷新检测结果。"""

    bug_detected: bool
    expectation_met: bool
    task_intent: str
    list_refreshed: bool
    target_region: str
    target_region_box: dict[str, int]
    roi_mean_abs_diff: float
    roi_changed_pixel_ratio: float
    reason: str
    confidence: float | None
    raw_response: str
    timing: dict[str, float] | None = None
    preprocess_evidence: dict[str, Any] | None = None


class ListRefreshDetector:
    """基于控件 bounds + 前后截图的列表刷新检测器。"""

    def __init__(
        self,
        evaluator: VisionEvaluator,
        logger: logging.Logger | None = None,
        debug: bool = False,
        crops_dir: Path | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.logger = logger or logging.getLogger("vision_gui_agent")
        self.debug = debug
        self.crops_dir = crops_dir

    @staticmethod
    def _read_image(path: Path) -> Any:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {path}")
        return image

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
    def _infer_list_region(
        image_width: int,
        image_height: int,
        control_bounds: ControlBounds,
    ) -> tuple[dict[str, int], str]:
        control_mid_y = control_bounds.y + (control_bounds.height / 2.0)
        top_height = max(0, control_bounds.y)
        bottom_height = max(0, image_height - (control_bounds.y + control_bounds.height))

        if control_mid_y <= image_height * 0.45:
            direction = "below"
        elif control_mid_y >= image_height * 0.65:
            direction = "above"
        else:
            direction = "below" if bottom_height >= top_height else "above"

        if direction == "below":
            y1 = min(image_height - 1, max(0, control_bounds.y + control_bounds.height))
            y2 = image_height
            region_desc = "控件下方列表区域"
        else:
            y1 = 0
            y2 = max(1, min(image_height, control_bounds.y))
            region_desc = "控件上方列表区域"

        # 兜底：推断区域过小则退化为全屏（避免误切导致证据不足）。
        if y2 - y1 < max(40, int(image_height * 0.15)):
            y1, y2 = 0, image_height
            region_desc = "全屏内容区域（推断列表区域过小，自动退化）"

        return {"x1": 0, "y1": y1, "x2": image_width, "y2": y2}, region_desc

    def _save_roi_artifacts(
        self,
        before_roi: Any,
        after_roi: Any,
        diff_gray: Any,
        task_id: str,
    ) -> list[Path]:
        if self.crops_dir is None:
            return []
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for image, name in [
            (before_roi, f"{task_id}_pp_list_before_roi.png"),
            (after_roi, f"{task_id}_pp_list_after_roi.png"),
            (diff_gray, f"{task_id}_pp_list_diff_gray.png"),
        ]:
            out_path = self.crops_dir / name
            if cv2.imwrite(str(out_path), image):
                paths.append(out_path)
        return paths

    def detect(
        self,
        before_image: Path,
        after_image: Path,
        control_bounds: ControlBounds,
        expected_list_refresh: bool = True,
        control_name_hint: str | None = None,
        task_id: str = "list_refresh_task",
    ) -> ListRefreshResult:
        """检测列表区域在控件响应后是否发生刷新。"""
        if not before_image.exists() or not after_image.exists():
            raise FileNotFoundError("输入前后页面截图不存在")
        control_bounds.validate()
        detect_start = time.perf_counter()

        before = self._read_image(before_image)
        after = self._read_image(after_image)
        h, w = before.shape[:2]
        if after.shape[:2] != (h, w):
            raise ValueError("前后截图分辨率不一致，无法执行列表刷新检测")

        list_region, list_region_desc = self._infer_list_region(
            image_width=w,
            image_height=h,
            control_bounds=control_bounds,
        )
        x1, y1, x2, y2 = list_region["x1"], list_region["y1"], list_region["x2"], list_region["y2"]
        before_roi = before[y1:y2, x1:x2]
        after_roi = after[y1:y2, x1:x2]
        roi_mean_abs_diff, roi_changed_pixel_ratio, diff_gray = self._calc_diff_metrics(before_roi, after_roi)
        preprocess_extra_images = self._save_roi_artifacts(before_roi, after_roi, diff_gray, task_id=task_id)

        preprocess_structured = {
            "evidence_type": "list_refresh_roi_diff",
            "list_region_desc": list_region_desc,
            "list_region_box": list_region,
            "roi_mean_abs_diff": round(roi_mean_abs_diff, 3),
            "roi_changed_pixel_ratio": round(roi_changed_pixel_ratio, 4),
            "notes": "该指标用于衡量目标列表区域在前后截图中的变化强度，辅助判定是否已刷新。",
        }
        preprocess_summary = (
            "前处理证据(list_refresh): "
            f"region={list_region_desc}, box=({x1},{y1})-({x2},{y2}), "
            f"mean_abs_diff={roi_mean_abs_diff:.3f}, changed_ratio={roi_changed_pixel_ratio:.4f}"
        )
        prompt_pack = build_prompt_for_type(
            task_type="list_refresh",
            context={
                "expected_list_refresh": expected_list_refresh,
                "control_name_hint": control_name_hint or "未提供",
                "control_bounds_x": control_bounds.x,
                "control_bounds_y": control_bounds.y,
                "control_bounds_width": control_bounds.width,
                "control_bounds_height": control_bounds.height,
                "preprocess_summary": preprocess_summary,
                "preprocess_structured": preprocess_structured,
            },
        )
        if self.debug:
            self.logger.info(
                "ListRefresh 调试: task_id=%s, expected_list_refresh=%s, list_region=%s",
                task_id,
                expected_list_refresh,
                list_region,
            )

        t_vlm_start = time.perf_counter()
        eval_result = self.evaluator.evaluate_json(
            before_image=before_image,
            after_image=after_image,
            system_prompt=prompt_pack.system_prompt,
            user_prompt=prompt_pack.user_prompt,
            task_id=task_id,
            required_fields={
                "list_refreshed": bool,
                "target_region": str,
                "reason": str,
            },
            extra_image_paths=preprocess_extra_images,
        )
        vlm_elapsed_ms = (time.perf_counter() - t_vlm_start) * 1000.0
        parsed = eval_result.parsed_json
        list_refreshed = bool(parsed.get("list_refreshed", False))
        expectation_met_raw = parsed.get("expectation_met")
        expectation_met = (
            bool(expectation_met_raw)
            if isinstance(expectation_met_raw, bool)
            else (list_refreshed == bool(expected_list_refresh))
        )
        bug_detected = not expectation_met

        return ListRefreshResult(
            bug_detected=bug_detected,
            expectation_met=expectation_met,
            task_intent=prompt_pack.task_intent,
            list_refreshed=list_refreshed,
            target_region=str(parsed.get("target_region", list_region_desc)),
            target_region_box=list_region,
            roi_mean_abs_diff=round(roi_mean_abs_diff, 3),
            roi_changed_pixel_ratio=round(roi_changed_pixel_ratio, 4),
            reason=str(parsed.get("reason", "")),
            confidence=float(parsed["confidence"]) if parsed.get("confidence") is not None else None,
            raw_response=eval_result.raw_response,
            timing={
                "detect_elapsed_ms": round((time.perf_counter() - detect_start) * 1000.0, 2),
                "vlm_elapsed_ms": round(vlm_elapsed_ms, 2),
            },
            preprocess_evidence=preprocess_structured,
        )
