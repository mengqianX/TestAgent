from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPORT_PATH_PATTERN = re.compile(r"测试报告已生成:\s*(.+)$", re.MULTILINE)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _extract_expected_passed(payload: dict[str, Any]) -> bool:
    """
    团队统一约定：expected_passed=true 表示功能正常（无 bug）。
    """
    if "expected_passed" in payload:
        return bool(payload["expected_passed"])
    if "groundtruth" in payload and isinstance(payload["groundtruth"], bool):
        return bool(payload["groundtruth"])
    if "groundtruth" in payload and isinstance(payload["groundtruth"], dict):
        gt = payload["groundtruth"]
        if "expected_passed" in gt:
            return bool(gt["expected_passed"])
    raise ValueError("case json 缺少 expected_passed 字段")


def _run_main(python_bin: str, input_file: Path, repo_root: Path) -> tuple[int, str]:
    cmd = [python_bin, "main.py", "--input-file", str(input_file)]
    env = os.environ.copy()
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(key, None)
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)
    return_code = proc.wait()
    output = "".join(output_lines)
    return return_code, output


def _resolve_report_path(output: str) -> Path:
    matches = REPORT_PATH_PATTERN.findall(output)
    if not matches:
        raise RuntimeError("运行输出中未找到报告路径（测试报告已生成）")
    return Path(matches[-1].strip())


def _resolve_task_pass(report: dict[str, Any]) -> tuple[bool, str]:
    task_type = str(report.get("input", {}).get("task_type", "")).strip().lower()
    for seg in report.get("segment_results", []):
        if not isinstance(seg, dict):
            continue
        if task_type in {"toast", "toast_validation", "toast_content"} and seg.get("detector") == "toast_detector":
            return (not bool(seg.get("bug_detected", False))), "toast.bug_detected"
        if task_type == "count_change" and seg.get("detector") == "count_change_detector":
            return (not bool(seg.get("bug_detected", False))), "count_change.bug_detected"
    bug_detected = bool(report.get("video_level_result", {}).get("bug_detected", False))
    return (not bug_detected), "video_level_result.bug_detected"


def _collect_toast_case_paths() -> list[Path]:
    return sorted(REPO_ROOT.glob("toast_*.json"), key=lambda p: p.name)


@pytest.fixture(scope="session", autouse=True)
def _generate_recent_stats_after_session():
    """
    在该 pytest 进程测试结束后自动生成最近 runs 统计表。
    """
    yield
    cmd = [sys.executable, "generate_recent_run_stats.py", "--limit", "30"]
    env = os.environ.copy()
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(key, None)
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        print(completed.stdout, end="")
    except Exception as exc:
        # 统计脚本失败不影响核心回归断言，只输出提示便于排查。
        print(f"[WARN] 自动生成统计表失败: {exc}")


@pytest.mark.parametrize("case_path", _collect_toast_case_paths(), ids=lambda p: p.name)
def test_toast_case_expected_passed_matches_model_output(case_path: Path):
    """
    回归测试核心断言：
    assert 模型输出(任务级 pass) == expected_passed

    """
    if not case_path.exists():
        pytest.fail(f"case 文件不存在: {case_path}")

    payload = json.loads(case_path.read_text(encoding="utf-8"))
    expected_passed = _extract_expected_passed(payload)
    code, output = _run_main(
        python_bin=sys.executable,
        input_file=case_path,
        repo_root=REPO_ROOT,
    )
    assert code == 0, f"main.py exited with code={code}\n{output}"

    report_path = _resolve_report_path(output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    actual_passed, source = _resolve_task_pass(report)

    assert actual_passed == expected_passed, (
        f"source={source}, expected_passed={expected_passed}, actual_passed={actual_passed}, "
        f"report={report_path}"
    )
