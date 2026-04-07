"""Prompt 构造器：使用 Python 逻辑按任务类型生成提示词。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class PromptPack:
    """一次模型调用需要的 system/user prompt。"""

    selected_prompt_type: str
    task_intent: str
    system_prompt: str
    user_prompt: str


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


PROMPT_BUILDERS: dict[str, Callable[[dict[str, Any]], PromptPack]] = {
    "general": _build_general_prompt,
    "loading": _build_loading_prompt,
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
