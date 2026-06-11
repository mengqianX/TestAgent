from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _collect_case_paths(case_set: str) -> list[Path]:
    pattern = f"{case_set}_*.json"
    return sorted(REPO_ROOT.glob(pattern), key=lambda p: p.name)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--case-set",
        action="store",
        default=os.getenv("VGA_CASE_SET", "toast"),
        choices=["toast", "general"],
        help="选择回归测试集前缀：toast 或 general（默认 toast，可由 VGA_CASE_SET 覆盖）",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case_path" not in metafunc.fixturenames:
        return
    case_set = str(metafunc.config.getoption("case_set")).strip().lower()
    case_paths = _collect_case_paths(case_set=case_set)
    if not case_paths:
        raise pytest.UsageError(f"未找到测试集文件：{case_set}_*.json")
    metafunc.parametrize("case_path", case_paths, ids=lambda p: p.name)


# 运行： python -m pytest tests/test_regression_live.py --collect-only -q --case-set general