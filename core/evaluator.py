"""评估模块：将关键帧和测试意图发送给 GPT-4o 进行 Bug 判定。"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Any

from openai import OpenAI


@dataclass
class EvaluationResult:
    """VLM 评估结果。"""

    bug_detected: bool
    reason: str
    raw_response: str


@dataclass
class JsonEvaluationResult:
    """通用 JSON 评估结果。"""

    parsed_json: dict[str, Any]
    raw_response: str


@dataclass
class TokenUsageTotals:
    """运行期累计 token 统计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class VisionEvaluator:
    """封装与 GPT-4o-vision 的交互逻辑。"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        logger: logging.Logger | None = None,
        debug: bool = False,
        log_full_data_url: bool = False,
        prompt_log_path: Path | None = None,
        prompt_text_path: Path | None = None,
    ) -> None:
        """
        初始化评估器。

        Args:
            api_key: OpenAI API Key。
            model: 使用的视觉模型名称。
            logger: 日志对象。
            debug: 是否输出调试日志。
            log_full_data_url: 是否打印完整 data URL（默认 False，避免日志过大）。
            prompt_log_path: 专用 Prompt 日志文件路径（JSONL）。
            prompt_text_path: 完整 Prompt 文本输出路径（单文件，.txt）。
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
        self.model = model or os.getenv("OPENAI_MODEL")
        self.logger = logger or logging.getLogger("vision_gui_agent")
        self.debug = debug
        self.log_full_data_url = log_full_data_url
        self.prompt_log_path = prompt_log_path
        self.prompt_text_path = prompt_text_path
        self._token_totals = TokenUsageTotals()
        self._prompt_call_count = 0
        self._calls_with_usage = 0

    def _append_prompt_log(self, record: dict[str, Any]) -> None:
        """
        追加写入 VLM 请求日志（JSONL）。

        Args:
            record: 单次请求记录。
        """
        if not self.prompt_log_path:
            return
        self.prompt_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.prompt_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _save_prompt_text(
        self,
        task_id: str,
        full_prompt: str,
        model_response_raw: str,
        token_usage: dict[str, Any] | None = None,
    ) -> Path | None:
        """
        保存单次请求的完整可读文本，便于人工调试。

        Args:
            task_id: 任务 ID。
            full_prompt: 完整提示词文本。
            model_response_raw: 模型原始输出。

        Returns:
            Path | None: 输出文本路径；未配置路径则为 None。
        """
        if not self.prompt_text_path:
            return None

        self.prompt_text_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "\n\n========================================\n"
            f"task_id={task_id}\n"
            f"timestamp={datetime.now().isoformat(timespec='seconds')}\n"
            "========================================\n"
            "===== FULL PROMPT =====\n"
            f"{full_prompt}\n\n"
            "===== MODEL RAW RESPONSE =====\n"
            f"{model_response_raw}\n"
        )
        if token_usage is not None:
            content += (
                "\n===== TOKEN USAGE =====\n"
                f"{json.dumps(token_usage, ensure_ascii=False)}\n"
            )
        with self.prompt_text_path.open("a", encoding="utf-8") as f:
            f.write(content)
        return self.prompt_text_path

    def _save_prompt_error_text(self, task_id: str, full_prompt: str, error_message: str) -> Path | None:
        """
        在请求失败时也落盘完整 prompt，便于排查。
        """
        if not self.prompt_text_path:
            return None
        self.prompt_text_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "\n\n========================================\n"
            f"task_id={task_id}\n"
            f"timestamp={datetime.now().isoformat(timespec='seconds')}\n"
            "========================================\n"
            "===== FULL PROMPT =====\n"
            f"{full_prompt}\n\n"
            "===== REQUEST ERROR =====\n"
            f"{error_message}\n"
        )
        with self.prompt_text_path.open("a", encoding="utf-8") as f:
            f.write(content)
        return self.prompt_text_path

    @staticmethod
    def _extract_token_usage(response: Any) -> dict[str, Any]:
        """
        从 OpenAI 响应中提取 token 统计。
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return {
                "available": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        prompt_tokens_raw = getattr(usage, "prompt_tokens", None)
        completion_tokens_raw = getattr(usage, "completion_tokens", None)
        total_tokens_raw = getattr(usage, "total_tokens", None)
        available = any(x is not None for x in (prompt_tokens_raw, completion_tokens_raw, total_tokens_raw))

        prompt_tokens = int(prompt_tokens_raw or 0)
        completion_tokens = int(completion_tokens_raw or 0)
        total_tokens = int(total_tokens_raw or (prompt_tokens + completion_tokens))

        return {
            "available": available,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _accumulate_token_usage(self, token_usage: dict[str, Any]) -> None:
        """
        累加单次请求的 token 数据。
        """
        if not bool(token_usage.get("available")):
            return
        self._token_totals.prompt_tokens += int(token_usage.get("prompt_tokens", 0))
        self._token_totals.completion_tokens += int(token_usage.get("completion_tokens", 0))
        self._token_totals.total_tokens += int(token_usage.get("total_tokens", 0))
        self._calls_with_usage += 1

    def get_token_usage_summary(self) -> dict[str, int]:
        """
        获取当前运行累计 token 统计。
        """
        return {
            "prompt_call_count": self._prompt_call_count,
            "calls_with_usage": self._calls_with_usage,
            "calls_without_usage": max(self._prompt_call_count - self._calls_with_usage, 0),
            "total_prompt_tokens": self._token_totals.prompt_tokens,
            "total_completion_tokens": self._token_totals.completion_tokens,
            "total_tokens": self._token_totals.total_tokens,
        }

    @staticmethod
    def _encode_image_to_data_url(image_path: Path) -> str:
        """
        将本地图片编码为 data URL，便于直接发送给 OpenAI。

        Args:
            image_path: 图片路径。

        Returns:
            str: data URL 字符串。
        """
        mime_type = "image/png"
        if image_path.suffix.lower() in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"

        with image_path.open("rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{image_b64}"

    def evaluate(
        self,
        before_image: Path,
        after_image: Path,
        test_intent: str,
        task_id: str = "unknown_task",
        system_prompt_override: str | None = None,
        user_prompt_override: str | None = None,
    ) -> EvaluationResult:
        """
        基于前后关键帧进行视觉 Bug 判定。

        Args:
            before_image: 交互前关键帧。
            after_image: 交互后关键帧。
            test_intent: 测试意图描述。
            task_id: 当前任务 ID（用于调试日志索引）。
            system_prompt_override: 自定义 system prompt。
            user_prompt_override: 自定义 user prompt。

        Returns:
            EvaluationResult: 标准化评估结果。

        Raises:
            ValueError: 模型输出不是合法 JSON 或缺少必填字段。
        """
        system_prompt = system_prompt_override or (
            "你是资深 GUI 自动化测试专家。"
            "你将收到交互前后的两张截图和测试意图。"
            "请严格输出 JSON，且只输出 JSON，不要包含任何额外文本。"
            "JSON 必须包含字段：bug_detected(bool)、reason(str)。"
        )
        user_prompt = user_prompt_override or (
            f"测试意图：{test_intent}\n"
            "请对比两张图判断是否存在视觉或交互结果相关的 Bug。"
            "如果存在明显异常（例如未跳转、错误弹窗、布局错乱、白屏、关键控件消失），"
            "bug_detected=true；否则为 false。"
        )
        generic_result = self.evaluate_json(
            before_image=before_image,
            after_image=after_image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task_id=task_id,
            required_fields={"bug_detected": bool, "reason": str},
        )
        parsed = generic_result.parsed_json
        return EvaluationResult(
            bug_detected=parsed["bug_detected"],
            reason=parsed["reason"],
            raw_response=generic_result.raw_response,
        )

    def evaluate_json(
        self,
        before_image: Path,
        after_image: Path,
        system_prompt: str,
        user_prompt: str,
        task_id: str = "unknown_task",
        required_fields: dict[str, type] | None = None,
        extra_image_paths: list[Path] | None = None,
        image_role_labels: list[str] | None = None,
    ) -> JsonEvaluationResult:
        """
        发送视觉请求并返回通用 JSON 结果（供高级场景复用）。

        Args:
            before_image: 主图-前。
            after_image: 主图-后。
            system_prompt: system prompt。
            user_prompt: user prompt。
            task_id: 任务 ID。
            required_fields: 期望字段及类型，例如 {"bug_detected": bool}。
            extra_image_paths: 额外图片列表（例如控件裁剪图）。
            image_role_labels: 与 all_images 对齐的图片角色文本标签，用于增强多图语义绑定。

        Returns:
            JsonEvaluationResult: 原始文本与解析后的 JSON。
        """
        if not before_image.exists():
            raise FileNotFoundError(f"交互前关键帧不存在: {before_image}")
        if not after_image.exists():
            raise FileNotFoundError(f"交互后关键帧不存在: {after_image}")
        extra_image_paths = extra_image_paths or []
        for image_path in extra_image_paths:
            if not image_path.exists():
                raise FileNotFoundError(f"额外图片不存在: {image_path}")

        all_images: list[Path] = [before_image, after_image] + extra_image_paths
        self.logger.info(
            "VLM调用开始: task_id=%s, model=%s, image_count=%s",
            task_id,
            self.model,
            len(all_images),
        )
        t_encode_start = time.perf_counter()
        data_urls: list[str] = [self._encode_image_to_data_url(p) for p in all_images]
        encode_elapsed_ms = (time.perf_counter() - t_encode_start) * 1000.0

        if image_role_labels is None:
            image_role_labels = ["图1:交互前关键帧", "图2:交互后关键帧"] + [
                f"图{idx + 3}:额外证据图" for idx in range(len(extra_image_paths))
            ]
        if len(image_role_labels) != len(all_images):
            raise ValueError(
                "image_role_labels 数量与图片数量不一致: "
                f"labels={len(image_role_labels)}, images={len(all_images)}"
            )

        content_for_api: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for label, data_url in zip(image_role_labels, data_urls):
            content_for_api.append({"type": "text", "text": label})
            content_for_api.append({"type": "image_url", "image_url": {"url": data_url}})
        messages_for_api: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_for_api},
        ]

        data_for_log: list[str] = [
            data_url if self.log_full_data_url else f"{data_url[:200]}..." for data_url in data_urls
        ]
        full_prompt_parts = [f"[SYSTEM]\n{system_prompt}", f"[USER]\n{user_prompt}"]
        image_labels = [f"IMAGE_{idx}_{label}" for idx, label in enumerate(image_role_labels)]
        for label, data in zip(image_labels, data_for_log):
            full_prompt_parts.append(f"[{label}]\n{data}")
        full_prompt = "\n\n".join(full_prompt_parts)

        prompt_record: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task_id": task_id,
            "model": self.model,
            "before_image": str(before_image),
            "after_image": str(after_image),
            "extra_images": [str(p) for p in extra_image_paths],
            "full_prompt": full_prompt,
            "messages": messages_for_api,
            "data_url_lengths": [len(x) for x in data_urls],
        }

        if self.debug:
            self.logger.info("===== VLM 请求调试信息开始 =====")
            self.logger.info("model=%s, before_image=%s, after_image=%s", self.model, before_image, after_image)
            self.logger.info("system_prompt=%s", system_prompt)
            self.logger.info("user_prompt=%s", user_prompt)
            self.logger.info("image_count=%s, data_url_lengths=%s", len(data_urls), [len(x) for x in data_urls])
            for idx, data_url in enumerate(data_urls):
                if self.log_full_data_url:
                    self.logger.info("image_%s_data_url=%s", idx, data_url)
                else:
                    self.logger.info("image_%s_data_url_prefix=%s...", idx, data_url[:120])
            self.logger.info("===== VLM 请求调试信息结束 =====")

        self._prompt_call_count += 1
        t_api_start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=messages_for_api,
                temperature=0,
            )
        except Exception as exc:
            api_elapsed_ms = (time.perf_counter() - t_api_start) * 1000.0
            error_text = f"{type(exc).__name__}: {exc}"
            self.logger.error(
                "VLM调用失败: task_id=%s, encode=%.2fms, api=%.2fms, error=%s",
                task_id,
                encode_elapsed_ms,
                api_elapsed_ms,
                error_text,
            )
            prompt_text_path = self._save_prompt_error_text(
                task_id=task_id,
                full_prompt=full_prompt,
                error_message=error_text,
            )
            prompt_record["request_error"] = error_text
            prompt_record["encode_elapsed_ms"] = round(encode_elapsed_ms, 2)
            prompt_record["api_elapsed_ms"] = round(api_elapsed_ms, 2)
            prompt_record["token_usage"] = {
                "available": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            prompt_record["prompt_text_file"] = str(prompt_text_path) if prompt_text_path else None
            self._append_prompt_log(prompt_record)
            raise

        api_elapsed_ms = (time.perf_counter() - t_api_start) * 1000.0
        token_usage = self._extract_token_usage(response)
        self._accumulate_token_usage(token_usage)
        raw_text: str = response.choices[0].message.content or "{}"
        prompt_text_path = self._save_prompt_text(
            task_id=task_id,
            full_prompt=full_prompt,
            model_response_raw=raw_text,
            token_usage=token_usage,
        )
        prompt_record["model_response_raw"] = raw_text
        prompt_record["encode_elapsed_ms"] = round(encode_elapsed_ms, 2)
        prompt_record["api_elapsed_ms"] = round(api_elapsed_ms, 2)
        prompt_record["token_usage"] = token_usage
        prompt_record["prompt_text_file"] = str(prompt_text_path) if prompt_text_path else None
        self._append_prompt_log(prompt_record)
        self.logger.info(
            "VLM调用结束: task_id=%s, encode=%.2fms, api=%.2fms, images=%s, tokens(prompt=%s, completion=%s, total=%s)",
            task_id,
            encode_elapsed_ms,
            api_elapsed_ms,
            len(data_urls),
            token_usage["prompt_tokens"],
            token_usage["completion_tokens"],
            token_usage["total_tokens"],
        )
        if self.debug:
            self.logger.info("VLM 原始输出: %s", raw_text)
            self.logger.info(
                "VLM耗时(task_id=%s): encode=%.2fms, api=%.2fms, images=%s",
                task_id,
                encode_elapsed_ms,
                api_elapsed_ms,
                len(data_urls),
            )

        try:
            parsed: dict[str, Any] = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"模型输出不是合法 JSON: {raw_text}") from exc

        if required_fields:
            for field, expected_type in required_fields.items():
                if field not in parsed:
                    raise ValueError(f"模型输出缺少必需字段: {field} | {parsed}")
                if not isinstance(parsed[field], expected_type):
                    raise ValueError(f"字段类型不匹配: {field} 期望 {expected_type} 实际 {type(parsed[field])}")

        return JsonEvaluationResult(parsed_json=parsed, raw_response=raw_text)
