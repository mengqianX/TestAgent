"""生成 runs 最近 N 条 report 的统计表。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import Counter


@dataclass
class StatRow:
    run_id: str
    task_id: str
    version: str
    passed: str
    model: str
    total_tokens: int
    total_completion_tokens: int
    total_prompt_tokens: int
    model_response_result: str
    error_description: str
    failure_reason: str


def _extract_expected_passed(payload: dict[str, Any]) -> bool | None:
    if "expected_passed" in payload:
        return bool(payload["expected_passed"])
    if "expected_pass" in payload:
        return bool(payload["expected_pass"])
    groundtruth = payload.get("groundtruth")
    if isinstance(groundtruth, bool):
        return bool(groundtruth)
    if isinstance(groundtruth, dict):
        if "expected_passed" in groundtruth:
            return bool(groundtruth["expected_passed"])
        if "expected_pass" in groundtruth:
            return bool(groundtruth["expected_pass"])
    return None


def _find_case_json(repo_root: Path, task_id: str) -> Path | None:
    candidates = [
        repo_root / f"{task_id}.json",
        repo_root / f"request_{task_id}.json",
        repo_root / "jsons" / f"{task_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_recent_reports(runs_dir: Path, limit: int) -> list[Path]:
    reports = sorted(
        runs_dir.glob("*/reports/report_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return reports[:limit]


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract_primary_segment(report: dict[str, Any]) -> dict[str, Any]:
    segment_results = report.get("segment_results")
    if isinstance(segment_results, list) and segment_results:
        first = segment_results[0]
        if isinstance(first, dict):
            return first
    return {}


def _normalize_text(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _compact_text(value: str, max_len: int = 80) -> str:
    clean = " ".join(value.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def _escape_md_cell(value: str, max_len: int | None = 120) -> str:
    content = value if max_len is None else _compact_text(value, max_len=max_len)
    return content.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _infer_failure_reason(expected_passed: bool | None, actual_passed: bool, error_description: str) -> str:
    if expected_passed is None:
        return "缺少expected_passed"
    if expected_passed == actual_passed:
        return "-"
    if expected_passed and (not actual_passed):
        if "不一致" in error_description or "语义" in error_description:
            return "误报-语义不一致"
        return "误报-模型判定失败"
    if (not expected_passed) and actual_passed:
        return "漏报-模型判定通过"
    return "判定不一致-其他"


def _build_rows(repo_root: Path, report_paths: list[Path], version: str, model: str) -> list[StatRow]:
    rows: list[StatRow] = []
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        run_id = str(report.get("run_id", ""))
        task_id = str(report.get("input", {}).get("task_id", ""))
        bug_detected = bool(report.get("video_level_result", {}).get("bug_detected", False))
        actual_passed = not bug_detected
        primary_segment = _extract_primary_segment(report)
        raw_model_response = _normalize_text(primary_segment.get("raw_model_response"), default="-")
        segment_reason = _normalize_text(primary_segment.get("reason"), default="-")

        case_json = _find_case_json(repo_root=repo_root, task_id=task_id)
        expected_passed: bool | None = None
        if case_json:
            case_payload = json.loads(case_json.read_text(encoding="utf-8"))
            expected_passed = _extract_expected_passed(case_payload)
        passed = "Yes" if expected_passed is not None and actual_passed == expected_passed else "No"
        error_description = segment_reason if passed == "No" else "-"
        failure_reason = _infer_failure_reason(
            expected_passed=expected_passed,
            actual_passed=actual_passed,
            error_description=segment_reason,
        )
        if passed != "No":
            failure_reason = "-"

        token_summary = report.get("token_usage_summary", {})
        if not isinstance(token_summary, dict):
            token_summary = {}
        model_response_result = raw_model_response
        rows.append(
            StatRow(
                run_id=run_id,
                task_id=task_id,
                version=version,
                passed=passed,
                model=model,
                total_tokens=_to_int(token_summary.get("total_tokens")),
                total_completion_tokens=_to_int(token_summary.get("total_completion_tokens")),
                total_prompt_tokens=_to_int(token_summary.get("total_prompt_tokens")),
                model_response_result=model_response_result,
                error_description=error_description,
                failure_reason=failure_reason,
            )
        )
    return rows


def _render_markdown(rows: list[StatRow], limit: int, version: str, model: str) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    lines: list[str] = [
        f"# 最近{limit}条测试记录统计（runs）",
        "",
        f"- 生成时间: {now}",
        f"- 样本范围: `data/runs` 下按 report 修改时间倒序最近 {limit} 条",
        "- 通过判定: `actual_passed = not bug_detected`，与 case json 的 `expected_passed/expected_pass` 一致则 `Yes`",
        f"- 版本固定: `{version}`；模型固定: `{model}`",
        "",
        "| 测试运行ID | 测试用例 | 版本 | 通过 | 模型 | 总Token数 | 总Completion Tokens | 总Prompt Tokens | 模型响应结果 | 错误描述 | 失败原因 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.run_id} | {row.task_id} | {row.version} | {row.passed} | "
            f"{row.model} | {row.total_tokens} | {row.total_completion_tokens} | {row.total_prompt_tokens} | "
            f"{_escape_md_cell(row.model_response_result, max_len=None)} | {_escape_md_cell(row.error_description)} | {row.failure_reason} |"
        )
    yes_count = sum(1 for row in rows if row.passed == "Yes")
    failure_rows = [row for row in rows if row.passed == "No"]
    reason_counter = Counter(row.failure_reason for row in failure_rows if row.failure_reason and row.failure_reason != "-")
    lines.extend(
        [
            "",
            f"- 合计: {len(rows)} 条（Yes={yes_count}, No={len(rows) - yes_count}）",
        ]
    )
    if failure_rows:
        lines.extend(
            [
                "",
                "## 失败原因观察",
                "",
                "| 失败原因类别 | 次数 | 占失败比例 |",
                "| --- | --- | --- |",
            ]
        )
        total_failures = len(failure_rows)
        for reason, count in reason_counter.most_common():
            ratio = f"{(count / total_failures) * 100:.1f}%"
            lines.append(f"| {reason} | {count} | {ratio} |")
        lines.extend(
            [
                "",
                "- 优先关注高频类别，先从样本最多的失败原因做 prompt 或候选帧策略优化。",
                "- 对 `误报-语义不一致`，建议补充动作语义和预期 toast 的约束词，降低语义误判。",
                "- 对 `漏报-模型判定通过`，建议回看候选帧选择策略与阈值，避免漏检异常画面。",
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 runs 目录最近 N 条 report 的 token 和通过情况。")
    parser.add_argument("--limit", type=int, default=30, help="统计最近多少条 report，默认 30。")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="data/runs",
        help="runs 根目录，默认 data/runs。",
    )
    parser.add_argument("--version", type=str, default="v2", help="版本列值，默认 v2。")
    parser.add_argument("--model", type=str, default="gpt-4o", help="模型列值，默认 gpt-4o。")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（.md）。默认写入 data/runs/recent{limit}_report_stats_{version}.md",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = (repo_root / runs_dir).resolve()
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs 目录不存在: {runs_dir}")

    limit = max(int(args.limit), 1)
    report_paths = _load_recent_reports(runs_dir=runs_dir, limit=limit)
    rows = _build_rows(
        repo_root=repo_root,
        report_paths=report_paths,
        version=str(args.version),
        model=str(args.model),
    )

    output = args.output
    if output:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = (repo_root / output_path).resolve()
    else:
        output_path = runs_dir / f"recent{limit}_report_stats_{args.version}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_markdown(rows=rows, limit=limit, version=str(args.version), model=str(args.model)),
        encoding="utf-8",
    )

    yes_count = sum(1 for row in rows if row.passed == "Yes")
    print(f"写入完成: {output_path}")
    print(f"统计条数: {len(rows)} (Yes={yes_count}, No={len(rows) - yes_count})")


if __name__ == "__main__":
    main()
