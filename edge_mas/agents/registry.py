from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    step: int
    safe_name: str
    display_name: str
    responsibility: str
    output: str


AGENTS: list[AgentSpec] = [
    AgentSpec(1, "literature_agent", "文献调研智能体", "检索和整理边缘检测相关研究", "literature_records.csv、literature_summary.md"),
    AgentSpec(2, "data_manager_agent", "数据管理智能体", "管理图像、mask、预处理和 edge GT", "processed_images/、gt/、file_pairs.csv"),
    AgentSpec(3, "algorithm_designer_agent", "算法设计智能体", "设计公开边缘检测算法对比方案", "algorithm_plan.json、algorithm_plan.md"),
    AgentSpec(4, "experiment_executor_agent", "实验执行智能体", "调用代码批量运行算法并保存结果", "edges/、experiment_config.json、run_log.csv"),
    AgentSpec(5, "metric_evaluator_agent", "指标评价智能体", "计算 Precision、Recall、F1、IoU 等指标", "metrics.csv、metrics_summary.csv"),
    AgentSpec(6, "result_analyst_agent", "结果分析智能体", "解释噪声、断裂、粗边缘、漏检等现象", "analysis_report.md、agent_analysis.json"),
]

COORDINATOR = AgentSpec(0, "coordinator_agent", "总控规划智能体", "理解需求、分解任务、顺序调度其他智能体", "agent_logs.json、workflow_state.json")

AUTO_GEN_NAME_MAP: dict[str, str] = {COORDINATOR.safe_name: COORDINATOR.display_name}
AUTO_GEN_NAME_MAP.update({a.safe_name: a.display_name for a in AGENTS})

DISPLAY_TO_SAFE = {v: k for k, v in AUTO_GEN_NAME_MAP.items()}
