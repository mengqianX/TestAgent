"""Prompt 构造器与版本化提示词加载。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class PromptPack:
    """一次模型调用需要的 system/user prompt。"""

    selected_prompt_type: str
    task_intent: str
    system_prompt: str
    user_prompt: str


class _SafeFormatDict(dict[str, Any]):
    """格式化时保留未知占位符，避免 KeyError。"""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _build_general_prompt(context: dict[str, Any]) -> PromptPack:
    """通用检测提示词。"""
    task_intent = "检测当前片段是否存在视觉或交互异常。"
    system_prompt = (
        "你是资深 GUI 自动化测试专家。"
        "你将收到交互前后的两张截图和测试意图。"
        "请严格输出 JSON，且只输出 JSON，不要包含任何额外文本。"
        "JSON 必须包含字段：bug_detected(bool)、reason(str)。"
    )
    user_prompt = (
        f"任务类型：{context['task_type']}\n"
        f"测试意图：{task_intent}\n"
        f"片段区间：{context['before_timestamp_sec']:.2f}s -> {context['after_timestamp_sec']:.2f}s\n"
        "请对比两张图判断是否存在视觉或交互结果相关的 Bug。"
        "若存在明显异常（例如未跳转、错误弹窗、布局错乱、白屏、关键控件消失），"
        "bug_detected=true；否则为 false。"
    )
    return PromptPack(
        selected_prompt_type="general",
        task_intent=task_intent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _build_loading_prompt(context: dict[str, Any]) -> PromptPack:
    """长时间加载检测提示词。"""
    task_intent = "检测页面是否处于异常长时间加载状态（疑似卡死或无反馈）。"
    system_prompt = (
        "你是资深移动端 GUI 质量专家，专门识别加载状态异常。"
        "请严格输出 JSON，且只输出 JSON，不要包含任何额外文本。"
        "JSON 必须包含字段：bug_detected(bool)、reason(str)。"
    )
    user_prompt = (
        "任务类型：loading（长时间加载检测）\n"
        f"测试意图：{task_intent}\n"
        f"片段区间：{context['before_timestamp_sec']:.2f}s -> {context['after_timestamp_sec']:.2f}s\n"
        "请重点判断是否存在以下问题：\n"
        "1) 加载指示器长期不消失\n"
        "2) 页面内容长时间不变化，疑似卡死\n"
        "3) 明显转圈但无结果反馈\n"
        "若符合上述异常，bug_detected=true；否则 false。"
    )
    return PromptPack(
        selected_prompt_type="loading",
        task_intent=task_intent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _build_toast_prompt(context: dict[str, Any]) -> PromptPack:
    """Toast 提示词（代码内置，不依赖外部 JSON 文件）。"""
    del context
    task_intent = "检测动作触发后的 toast 文案是否与预期语义一致。"
    system_prompt = (
        "你是移动端UI测试助手。任务是判断候选时刻是否出现“瞬时toast/snackbar”，并校验其文案语义是否符合动作。"
        "请严格按顺序执行并遵守时序约束：\n"
        "1) 先判断 toast_visible；\n"
        "2) 仅当 toast_visible=true 时再提取 toast_text；\n"
        "3) action_semantic 只能依据图1(动作前)与图2(候选)推断，禁止根据图3或toast_text反推动作；\n"
        "4) inferred_expected_toast_text 只能依据 action_semantic 生成，不能直接照搬 toast_text；\n"
        "5) 若图1与图2仍是同一操作上下文(如同一弹窗/同一表单)，优先判定为该上下文对应动作，不得被图3中的toast语义牵引改判；\n"
        "6) 若动作属于低可观测手势（如左右滑动条目）且仅靠图1+图2无法稳定判断动作语义，必须输出 action_semantic=\"unknown\"，并将 expectation_met=null、reverse_inference_risk=\"high\"。\n"
        "固定页面文案、列表内容、标题栏、底部统计不算toast。若 toast_visible=false，必须输出 toast_text=\"\"、inferred_expected_toast_text=\"\"、expectation_met=false。"
        "请只输出 JSON，不要额外文本。"
        "JSON 必须包含字段：toast_visible(bool), toast_text(str), action_semantic(str), inferred_expected_toast_text(str), expectation_met(bool|null), reverse_inference_risk(str), action_evidence_from_frame12(str), toast_evidence_from_frame23(str), confidence(number, 0~1), reason(str)。"
        "其中 reverse_inference_risk 只能是 low 或 high。action_evidence_from_frame12 必须只引用图1和图2可见证据；toast_evidence_from_frame23 只引用图2和图3可见证据。reason 限制为一句话且不超过40个汉字。"
    )
    user_prompt = (
        "任务意图：{task_intent}\n"
        "候选帧时间：{candidate_timestamp_sec:.2f}s\n"
        "测试关键词提示（可选）：{keywords_text}\n"
        "前处理摘要：{preprocess_summary}\n"
        "前处理证据(JSON, 精简)：{preprocess_structured_json}\n"
        "请基于三张完整帧判断是否出现toast：图1=动作前完整帧，图2=候选完整帧，图3=候选后完整帧。\n"
        "关键规则：先由图1+图2确定 action_semantic，再判断图2/图3中的toast是否与该动作匹配；禁止使用图3的toast文本反推动作语义。若动作可观测性不足（典型是滑动手势），请输出 action_semantic=unknown 且 expectation_met=null。"
    )
    return PromptPack(
        selected_prompt_type="toast",
        task_intent=task_intent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def build_toast_prompt(
    prompt_version: str = "current",
    prompts_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> PromptPack:
    """构建 toast 提示词（兼容接口，内部使用代码内置模板）。"""
    del prompt_version, prompts_dir
    if isinstance(logger, logging.Logger):
        logger.info("toast prompt loaded: source=inline")
    return build_prompt_for_type(
        task_type="toast",
        context={},
    )


def render_toast_user_prompt(prompt_pack: PromptPack, context: dict[str, Any]) -> str:
    """将上下文填充到 toast user prompt 模板。"""
    return prompt_pack.user_prompt.format_map(_SafeFormatDict(context))


def _build_count_change_prompt(context: dict[str, Any]) -> PromptPack:
    """数量变化检测提示词。"""
    task_intent = "检测目标控件触发后，其语义关联的数量指标是否按预期变化。"
    metric_hints = context.get("metric_hints")
    hints_text = "、".join(metric_hints) if metric_hints else "无"
    control_name = context.get("control_name_hint") or "未提供"
    preprocess_structured = context.get("preprocess_structured")
    preprocess_structured_text = (
        json.dumps(preprocess_structured, ensure_ascii=False) if preprocess_structured else "{}"
    )
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
        f"期望变化规则：{context['expected_change']}（increase/decrease/any_change/no_change/any）\n"
        f"控件名称提示：{control_name}\n"
        f"指标关键词提示：{hints_text}\n"
        f"控件 bounds：x={context['control_bounds_x']}, y={context['control_bounds_y']}, "
        f"width={context['control_bounds_width']}, height={context['control_bounds_height']}\n"
        f"前处理摘要：{context['preprocess_summary']}\n"
        f"前处理结构化证据(JSON)：{preprocess_structured_text}\n"
        "请先识别该控件语义，再关联最相关数字指标，判断前后是否变化及方向。"
        "请优先使用上述证据来定位需要关注的区域，但最终结论以图像事实为准。"
    )
    return PromptPack(
        selected_prompt_type="count_change",
        task_intent=task_intent,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


PROMPT_BUILDERS: dict[str, Callable[[dict[str, Any]], PromptPack]] = {
    "general": _build_general_prompt,
    "loading": _build_loading_prompt,
    "toast": _build_toast_prompt,
    "count_change": _build_count_change_prompt,
}


def build_prompt_for_type(task_type: str, context: dict[str, Any]) -> PromptPack:
    """
    按 task_type 构造提示词，未知类型自动回退到 general。

    Args:
        task_type: 任务类型，例如 loading。
        context: 上下文变量。

    Returns:
        PromptPack: 构造完成的提示词对象。
    """
    normalized = (task_type or "").strip().lower()
    builder = PROMPT_BUILDERS.get(normalized, PROMPT_BUILDERS["general"])
    return builder(context)


def build_count_change_prompt(context: dict[str, Any]) -> PromptPack:
    """构建 count_change 提示词（兼容接口，内部走统一分发）。"""
    return build_prompt_for_type(task_type="count_change", context=context)
