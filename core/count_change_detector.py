"""数量变化检测模块：识别控件语义并关联对应数字变化。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2

from core.evaluator import VisionEvaluator
from core.preprocessor import GuiPreprocessor


@dataclass
class ControlBounds:
    """控件边界框（像素坐标）。"""

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ControlBounds":
        """
        从 JSON payload 读取 bounds。

        Args:
            payload: bounds 字典，支持 x/y/width/height。

        Returns:
            ControlBounds: 结构化边界框。
        """
        return cls(
            x=int(payload["x"]),
            y=int(payload["y"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )

    @classmethod
    def from_list(cls, payload: list[Any]) -> "ControlBounds":
        """
        从数组读取 bounds，格式为 [x1, y1, x2, y2]。
        """
        if len(payload) != 4:
            raise ValueError(f"bounds 数组长度必须为 4，当前为 {len(payload)}")
        x1, y1, x2, y2 = [int(v) for v in payload]
        width = x2 - x1
        height = y2 - y1
        return cls(x=x1, y=y1, width=width, height=height)

    def validate(self) -> None:
        """校验边界框是否合法。"""
        if self.x < 0 or self.y < 0:
            raise ValueError(f"bounds 的 x/y 不能小于 0: {self}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"bounds 的 width/height 必须大于 0: {self}")


@dataclass
class CountChangeResult:
    """数量变化检测结果。"""

    bug_detected: bool
    expectation_met: bool
    task_intent: str
    semantic_target: str
    linked_metric: str
    before_value: int | None
    after_value: int | None
    value_changed: bool
    change_direction: str
    reason: str
    confidence: float | None
    raw_response: str
    timing: dict[str, float] | None = None
    preprocess_evidence: dict[str, Any] | None = None


class CountChangeDetector:
    """基于多图语义关联的数量变化检测器。"""

    def __init__(
        self,
        evaluator: VisionEvaluator,
        logger: logging.Logger | None = None,
        debug: bool = False,
        crops_dir: Path | None = None,
        preprocessor: GuiPreprocessor | None = None,
        enable_preprocess: bool = True,
    ) -> None:
        self.evaluator = evaluator
        self.logger = logger or logging.getLogger("vision_gui_agent")
        self.debug = debug
        self.crops_dir = crops_dir
        self.preprocessor = preprocessor
        self.enable_preprocess = enable_preprocess

    def _crop_control(self, image_path: Path, bounds: ControlBounds, output_path: Path) -> Path:
        """
        从原图裁剪控件区域，便于辅助语义关联。
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"无法读取图片: {image_path}")
        h, w = image.shape[:2]
        x1 = max(0, min(bounds.x, w - 1))
        y1 = max(0, min(bounds.y, h - 1))
        x2 = max(x1 + 1, min(bounds.x + bounds.width, w))
        y2 = max(y1 + 1, min(bounds.y + bounds.height, h))
        crop = image[y1:y2, x1:x2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), crop):
            raise RuntimeError(f"裁剪图保存失败: {output_path}")
        return output_path

    @staticmethod
    def _compute_expectation_met(expected_change: str, value_changed: bool, direction: str) -> bool:
        """根据期望规则与模型判定计算 expectation_met。"""
        rule = expected_change.strip().lower()
        if rule == "increase":
            return direction == "increase"
        if rule == "decrease":
            return direction == "decrease"
        if rule == "no_change":
            return not value_changed
        if rule == "any":
            return True
        # 默认 any_change
        return value_changed

    def detect(
        self,
        before_image: Path,
        after_image: Path,
        control_bounds: ControlBounds,
        expected_change: str = "any_change",
        metric_hints: list[str] | None = None,
        control_name_hint: str | None = None,
        task_id: str = "count_change_task",
    ) -> CountChangeResult:
        """
        检测控件触发后关联数量是否变化。
        """
        if not before_image.exists() or not after_image.exists():
            raise FileNotFoundError("输入前后页面截图不存在")
        control_bounds.validate()
        detect_start = time.perf_counter()

        task_intent = "检测目标控件触发后，其语义关联的数量指标是否按预期变化。"
        hints_text = "、".join(metric_hints or []) if metric_hints else "无"
        control_name = control_name_hint or "未提供"

        before_crop_path: Path | None = None
        after_crop_path: Path | None = None
        preprocess_structured: dict[str, Any] | None = None
        preprocess_summary = "无"
        preprocess_extra_images: list[Path] = []
        preprocess_elapsed_ms = 0.0
        if self.crops_dir:
            before_crop_path = self._crop_control(
                before_image,
                control_bounds,
                self.crops_dir / f"{task_id}_before_control.png",
            )
            after_crop_path = self._crop_control(
                after_image,
                control_bounds,
                self.crops_dir / f"{task_id}_after_control.png",
            )
        if self.enable_preprocess and self.preprocessor is not None:
            try:
                t_pre_start = time.perf_counter()
                preprocess_evidence = self.preprocessor.prepare_count_change(
                    before_image=before_image,
                    after_image=after_image,
                    bounds={
                        "x": control_bounds.x,
                        "y": control_bounds.y,
                        "width": control_bounds.width,
                        "height": control_bounds.height,
                    },
                    task_id=task_id,
                )
                preprocess_structured = preprocess_evidence.structured
                preprocess_summary = preprocess_evidence.summary
                preprocess_extra_images = preprocess_evidence.extra_image_paths
                preprocess_elapsed_ms = (time.perf_counter() - t_pre_start) * 1000.0
                self.logger.info(
                    "count_change 前处理完成: task_id=%s, elapsed=%.2fms, extra_images=%s",
                    task_id,
                    preprocess_elapsed_ms,
                    len(preprocess_extra_images),
                )
            except Exception as exc:  # pylint: disable=broad-except
                if self.debug:
                    self.logger.warning("count_change 前处理失败: %s", exc)

        system_prompt = (
            "你是资深 GUI 自动化测试专家，擅长控件语义识别与跨区域数字关联。"
            "你将看到前后页面截图以及控件 bounds 信息。"
            "请识别控件语义，并找到与其逻辑关联的数字指标（可不在控件附近）。"
            "请严格输出 JSON，且只输出 JSON。"
            "JSON 必须包含字段："
            "semantic_target(str), linked_metric(str), before_value(int|null), after_value(int|null), "
            "value_changed(bool), change_direction(str), expectation_met(bool), confidence(number), reason(str)。"
        )
        user_prompt = (
            f"任务意图：{task_intent}\n"
            f"期望变化规则：{expected_change}（increase/decrease/any_change/no_change/any）\n"
            f"控件名称提示：{control_name}\n"
            f"指标关键词提示：{hints_text}\n"
            f"控件 bounds：x={control_bounds.x}, y={control_bounds.y}, width={control_bounds.width}, "
            f"height={control_bounds.height}\n"
            f"前处理摘要：{preprocess_summary}\n"
            f"前处理结构化证据(JSON)：{json.dumps(preprocess_structured, ensure_ascii=False) if preprocess_structured else '{}'}\n"
            "请先识别该控件语义，再关联最相关数字指标，判断前后是否变化及方向。"
            "请优先使用上述证据来定位需要关注的区域，但最终结论以图像事实为准。"
        )

        if self.debug:
            self.logger.info("CountChange 调试: task_id=%s, expected_change=%s", task_id, expected_change)
            self.logger.info("CountChange user_prompt=%s", user_prompt)

        extra_images: list[Path] = []
        if before_crop_path and after_crop_path:
            extra_images.extend([before_crop_path, after_crop_path])
        extra_images.extend(preprocess_extra_images)
        t_vlm_start = time.perf_counter()
        self.logger.info("count_change 开始调用VLM: task_id=%s, image_count=%s", task_id, 2 + len(extra_images))
        generic_result = self.evaluator.evaluate_json(
            before_image=before_image,
            after_image=after_image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task_id=task_id,
            required_fields={
                "semantic_target": str,
                "linked_metric": str,
                "value_changed": bool,
                "change_direction": str,
                "reason": str,
            },
            extra_image_paths=extra_images,
        )
        vlm_elapsed_ms = (time.perf_counter() - t_vlm_start) * 1000.0
        parsed = generic_result.parsed_json
        raw_text = generic_result.raw_response

        before_value = parsed.get("before_value")
        after_value = parsed.get("after_value")
        if before_value is not None:
            before_value = int(before_value)
        if after_value is not None:
            after_value = int(after_value)

        value_changed = bool(parsed.get("value_changed", False))
        direction = str(parsed.get("change_direction", "unknown")).lower()
        expectation_met = parsed.get("expectation_met")
        if not isinstance(expectation_met, bool):
            expectation_met = self._compute_expectation_met(
                expected_change=expected_change,
                value_changed=value_changed,
                direction=direction,
            )
        bug_detected = not expectation_met

        result = CountChangeResult(
            bug_detected=bug_detected,
            expectation_met=expectation_met,
            task_intent=task_intent,
            semantic_target=str(parsed.get("semantic_target", "")),
            linked_metric=str(parsed.get("linked_metric", "")),
            before_value=before_value,
            after_value=after_value,
            value_changed=value_changed,
            change_direction=direction,
            reason=str(parsed.get("reason", "")),
            confidence=float(parsed["confidence"]) if parsed.get("confidence") is not None else None,
            raw_response=raw_text,
            timing={
                "detect_elapsed_ms": round((time.perf_counter() - detect_start) * 1000.0, 2),
                "preprocess_elapsed_ms": round(preprocess_elapsed_ms, 2),
                "vlm_elapsed_ms": round(vlm_elapsed_ms, 2),
            },
            preprocess_evidence=preprocess_structured,
        )
        return result
