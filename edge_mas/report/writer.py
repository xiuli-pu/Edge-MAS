from __future__ import annotations

from pathlib import Path

import pandas as pd

from edge_mas.utils import read_text_if_exists


def dataframe_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "暂无数据。"
    return df.head(max_rows).to_markdown(index=False)


def build_research_report(
    task: str,
    run_dir: Path,
    selected_algorithms: list[str],
    literature_summary_path: Path | None,
    algorithm_plan_path: Path | None,
    metrics_path: Path | None,
    metrics_summary_path: Path | None,
    analysis_report_path: Path | None,
    agent_logs: list[dict],
) -> str:
    metrics = pd.DataFrame()
    summary = pd.DataFrame()
    if metrics_path and metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
    if metrics_summary_path and metrics_summary_path.exists():
        summary = pd.read_csv(metrics_summary_path)

    logs_lines = []
    for log in agent_logs:
        logs_lines.append(
            f"- **Step {log.get('step')} {log.get('agent')}**：输入：{log.get('input')}；输出：{log.get('output')}；说明：{log.get('message')}"
        )

    text = f"""# Edge-MAS 多智能体边缘检测实验报告

## 1. 研究任务

{task}

## 2. 多智能体协作流程

本系统将边缘检测实验流程拆分为六个自动执行阶段：文献调研、数据管理、算法设计、实验执行、指标评价和结果分析。各阶段由不同智能体负责，并通过共享文件与工作流状态传递阶段性产物。

{chr(10).join(logs_lines) if logs_lines else '暂无协作日志。'}

## 3. 文献调研智能体产物

{read_text_if_exists(literature_summary_path)}

## 4. 算法设计智能体产物

{read_text_if_exists(algorithm_plan_path)}

## 5. 实验设置

- 实验目录：`{run_dir}`
- 对比算法：{', '.join(selected_algorithms)}
- 边缘检测任务类型：裂缝图像边缘提取与算法对比

## 6. 指标评价结果

### 6.1 方法平均指标

{dataframe_markdown(summary)}

### 6.2 单图像指标明细

{dataframe_markdown(metrics)}

## 7. 结果分析智能体产物

{read_text_if_exists(analysis_report_path)}

## 8. 结论

本系统以多智能体协作为核心，将边缘检测科研任务从单一算法调用扩展为完整实验流程。用户输入研究任务和数据后，系统自动完成文献整理、数据预处理、算法方案生成、批量实验执行、定量指标评价和结果现象解释。该流程能够清晰体现不同智能体之间的分工、产物传递和顺序协作关系。
"""
    return text


def save_report(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
