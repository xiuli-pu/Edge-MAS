from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pandas as pd

from edge_mas.agents.autogen_runner import run_autogen_sync, save_autogen_conversation
from edge_mas.agents.registry import AGENTS, COORDINATOR
from edge_mas.config import ExperimentConfig, LLMConfig, RESULTS_DIR
from edge_mas.image.algorithms import mask_to_edge_gt, run_algorithms
from edge_mas.image.demo import generate_demo_dataset
from edge_mas.image.io_utils import imwrite, pair_by_stem, read_gray, save_pair_table
from edge_mas.image.metrics import edge_metrics_tolerance
from edge_mas.literature.manager import copy_literature_records_to_run, create_literature_summary
from edge_mas.report.writer import build_research_report, save_report
from edge_mas.utils import copy_files, make_run_dir, zip_directory


@dataclass
class WorkflowConfig:
    task: str
    selected_algorithms: list[str]
    exp_cfg: ExperimentConfig
    use_demo_data: bool = True
    image_paths: list[Path] = field(default_factory=list)
    mask_paths: list[Path] = field(default_factory=list)
    export_report: bool = True
    demo_count: int = 3
    llm_cfg: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class StepResult:
    step: int
    agent: str
    title: str
    status: str
    input: str
    output: str
    message: str
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass
class WorkflowState:
    task: str
    run_dir: Path
    selected_algorithms: list[str]
    exp_cfg: ExperimentConfig
    use_demo_data: bool
    image_paths: list[Path] = field(default_factory=list)
    mask_paths: list[Path] = field(default_factory=list)
    processed_image_paths: list[Path] = field(default_factory=list)
    gt_paths: list[Path] = field(default_factory=list)
    literature_csv_path: Path | None = None
    literature_jsonl_path: Path | None = None
    literature_summary_path: Path | None = None
    file_pairs_path: Path | None = None
    algorithm_plan_json_path: Path | None = None
    algorithm_plan_md_path: Path | None = None
    edges_dir: Path | None = None
    experiment_config_path: Path | None = None
    run_log_path: Path | None = None
    metrics_path: Path | None = None
    metrics_summary_path: Path | None = None
    analysis_report_path: Path | None = None
    agent_analysis_path: Path | None = None
    final_report_path: Path | None = None
    zip_path: Path | None = None
    autogen_conversation_json_path: Path | None = None
    autogen_conversation_md_path: Path | None = None
    autogen_rows: list[dict] = field(default_factory=list)
    agent_logs: list[dict] = field(default_factory=list)


def _log(state: WorkflowState, result: StepResult) -> None:
    state.agent_logs.append(asdict(result))
    path = state.run_dir / "agent_logs.json"
    path.write_text(json.dumps(state.agent_logs, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_state(state: WorkflowState) -> Path:
    def conv(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, ExperimentConfig):
            return asdict(obj)
        raise TypeError(type(obj))
    path = state.run_dir / "workflow_state.json"
    path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2, default=conv), encoding="utf-8")
    return path


def init_state(cfg: WorkflowConfig) -> WorkflowState:
    """初始化工作流状态，并强制先启动 AutoGen 多智能体协作。

    这一版本的多智能体思想明确基于 AutoGen：
    只有 AutoGen 成功生成总控规划和多智能体协作说明后，
    系统才会继续进入文献调研、数据管理、算法设计等工具执行阶段。
    """
    run_dir = make_run_dir(RESULTS_DIR)
    state = WorkflowState(
        task=cfg.task,
        run_dir=run_dir,
        selected_algorithms=cfg.selected_algorithms,
        exp_cfg=cfg.exp_cfg,
        use_demo_data=cfg.use_demo_data,
    )

    autogen_task = f"""
你们是一个基于 AutoGen 的边缘检测科研多智能体团队。
用户任务：{cfg.task}
数据来源：{'系统演示数据' if cfg.use_demo_data else '用户上传图像与 mask'}
选择算法：{', '.join(cfg.selected_algorithms)}

请按以下顺序进行协作说明，每个智能体只说明自己阶段的输入、处理、输出，以及输出如何交给下一阶段：
1. 总控规划智能体
2. 文献调研智能体
3. 数据管理智能体
4. 算法设计智能体
5. 实验执行智能体
6. 指标评价智能体
7. 结果分析智能体

注意：系统只使用公开传统边缘检测算法，不出现任何隐藏算法。
""".strip()

    rows = run_autogen_sync(
        task=autogen_task,
        model=cfg.llm_cfg.model,
        base_url=cfg.llm_cfg.base_url,
        api_key=cfg.llm_cfg.api_key,
        max_messages=8,
    )
    saved = save_autogen_conversation(rows, run_dir)
    state.autogen_rows = rows
    state.autogen_conversation_json_path = saved["json"]
    state.autogen_conversation_md_path = saved["md"]

    coord = StepResult(
        step=0,
        agent=COORDINATOR.display_name,
        title="AutoGen 总控规划",
        status="完成",
        input="用户研究任务、数据来源、算法选择、实验参数、AutoGen 模型配置",
        output="autogen_conversation.json、autogen_conversation.md、六阶段多智能体协作计划",
        message="AutoGen 已成功启动总控规划智能体和六个专业智能体，完成协作任务分解；系统随后按照 AutoGen 协作计划进入工具执行阶段。",
        artifacts={
            "run_dir": str(run_dir),
            "autogen_conversation.json": str(saved["json"]),
            "autogen_conversation.md": str(saved["md"]),
        },
    )
    _log(state, coord)
    _save_state(state)
    return state


def step1_literature_agent(state: WorkflowState) -> StepResult:
    copied = copy_literature_records_to_run(state.run_dir)
    summary_path = create_literature_summary(
        state.run_dir / "literature" / "literature_summary.md",
        task=state.task,
        selected_algorithms=state.selected_algorithms,
    )
    state.literature_csv_path = copied["csv"]
    state.literature_jsonl_path = copied["jsonl"]
    state.literature_summary_path = summary_path
    res = StepResult(
        step=1,
        agent="文献调研智能体",
        title="Step 1 文献调研",
        status="完成",
        input="研究任务、算法关键词、边缘检测主题词",
        output="literature_records.csv、literature_records.jsonl、literature_summary.md",
        message="文献调研智能体已保存边缘检测种子文献，并生成文献调研摘要，供算法设计智能体参考。",
        artifacts={
            "literature_records.csv": str(state.literature_csv_path),
            "literature_summary.md": str(summary_path),
        },
    )
    _log(state, res)
    _save_state(state)
    return res


def step2_data_manager_agent(state: WorkflowState, cfg: WorkflowConfig) -> StepResult:
    source_dir = state.run_dir / "source_data"
    if cfg.use_demo_data:
        imgs, masks = generate_demo_dataset(source_dir, n=cfg.demo_count)
    else:
        imgs = copy_files(cfg.image_paths, source_dir / "images")
        masks = copy_files(cfg.mask_paths, source_dir / "masks") if cfg.mask_paths else []

    pairs = pair_by_stem(imgs, masks)
    processed_dir = state.run_dir / "processed_images"
    gt_dir = state.run_dir / "gt"
    processed_paths: list[Path] = []
    gt_paths: list[Path] = []

    for img_path, mask_path in pairs.items():
        gray = read_gray(img_path)
        out_img = processed_dir / f"{img_path.stem}_gray.png"
        imwrite(out_img, gray)
        processed_paths.append(out_img)

        if mask_path is not None:
            mask = read_gray(mask_path)
            if mask.shape != gray.shape:
                mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt = mask_to_edge_gt(mask)
            gt_path = gt_dir / f"{img_path.stem}_edge_gt.png"
            imwrite(gt_path, gt)
            gt_paths.append(gt_path)

    pair_path = save_pair_table(pairs, state.run_dir / "file_pairs.csv")
    state.image_paths = imgs
    state.mask_paths = masks
    state.processed_image_paths = processed_paths
    state.gt_paths = gt_paths
    state.file_pairs_path = pair_path
    res = StepResult(
        step=2,
        agent="数据管理智能体",
        title="Step 2 数据管理",
        status="完成",
        input="原始图像、mask 图、数据来源设置",
        output="processed_images/、gt/、file_pairs.csv",
        message=f"数据管理智能体已完成 {len(processed_paths)} 张图像的灰度化和文件配对；生成 {len(gt_paths)} 个 edge GT。",
        artifacts={
            "file_pairs.csv": str(pair_path),
            "processed_images": str(processed_dir),
            "gt": str(gt_dir),
        },
    )
    _log(state, res)
    _save_state(state)
    return res


def _algorithm_explanation(name: str) -> str:
    return {
        "Sobel": "一阶梯度边缘算子，计算简单，适合作为传统梯度基线。",
        "Prewitt": "一阶差分边缘算子，与 Sobel 类似但卷积权重更均匀。",
        "LoG": "高斯平滑后使用拉普拉斯二阶导数寻找边缘，对细节和噪声较敏感。",
        "Canny": "多阶段边缘检测方法，包含平滑、梯度、非极大值抑制和双阈值连接。",
        "Morphological Gradient": "形态学梯度通过膨胀与腐蚀差分提取结构边界。",
    }.get(name, "公开边缘检测方法。")


def step3_algorithm_designer_agent(state: WorkflowState) -> StepResult:
    plan = {
        "task": state.task,
        "selected_algorithms": state.selected_algorithms,
        "fair_comparison_principles": [
            "所有算法使用同一批灰度图像作为输入。",
            "所有算法结果保存为二值边缘图。",
            "有 mask 时统一使用同一 edge GT 和相同容差 k 计算指标。",
            "实验参数保存到 experiment_config.json，保证可复现。",
        ],
        "algorithms": [
            {"name": name, "role": _algorithm_explanation(name)} for name in state.selected_algorithms
        ],
    }
    json_path = state.run_dir / "algorithm_plan.json"
    md_path = state.run_dir / "algorithm_plan.md"
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Step 3 算法设计智能体输出：算法对比方案", "", f"## 研究任务", state.task, "", "## 对比算法"]
    for item in plan["algorithms"]:
        lines.append(f"- **{item['name']}**：{item['role']}")
    lines.extend(["", "## 公平对比原则"])
    for p in plan["fair_comparison_principles"]:
        lines.append(f"- {p}")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    state.algorithm_plan_json_path = json_path
    state.algorithm_plan_md_path = md_path
    res = StepResult(
        step=3,
        agent="算法设计智能体",
        title="Step 3 算法设计",
        status="完成",
        input="研究任务、文献调研摘要、数据情况、用户选择算法",
        output="algorithm_plan.json、algorithm_plan.md",
        message="算法设计智能体已生成公开边缘检测算法对比方案，并明确公平对比原则。",
        artifacts={"algorithm_plan.json": str(json_path), "algorithm_plan.md": str(md_path)},
    )
    _log(state, res)
    _save_state(state)
    return res


def step4_experiment_executor_agent(state: WorkflowState) -> StepResult:
    edges_dir = state.run_dir / "edges"
    rows = []
    for img_path in state.processed_image_paths:
        gray = read_gray(img_path)
        results = run_algorithms(gray, state.exp_cfg, state.selected_algorithms)
        for method, edge in results.items():
            edge_path = edges_dir / method / f"{img_path.stem}_{method.replace(' ', '_')}.png"
            imwrite(edge_path, edge)
            rows.append({"image": img_path.name, "method": method, "edge_path": str(edge_path)})

    run_log_path = state.run_dir / "run_log.csv"
    pd.DataFrame(rows).to_csv(run_log_path, index=False, encoding="utf-8-sig")
    cfg_path = state.run_dir / "experiment_config.json"
    cfg_path.write_text(json.dumps(asdict(state.exp_cfg) | {"selected_algorithms": state.selected_algorithms}, ensure_ascii=False, indent=2), encoding="utf-8")
    state.edges_dir = edges_dir
    state.run_log_path = run_log_path
    state.experiment_config_path = cfg_path
    res = StepResult(
        step=4,
        agent="实验执行智能体",
        title="Step 4 实验执行",
        status="完成",
        input="algorithm_plan.json、processed_images/、实验参数",
        output="edges/、experiment_config.json、run_log.csv",
        message=f"实验执行智能体已运行 {len(state.selected_algorithms)} 种算法，生成 {len(rows)} 张边缘检测结果图。",
        artifacts={"edges": str(edges_dir), "experiment_config.json": str(cfg_path), "run_log.csv": str(run_log_path)},
    )
    _log(state, res)
    _save_state(state)
    return res


def _find_gt_for_image(state: WorkflowState, processed_name: str) -> Path | None:
    stem = processed_name.replace("_gray.png", "")
    for gt in state.gt_paths:
        if gt.name.startswith(stem):
            return gt
    return None


def step5_metric_evaluator_agent(state: WorkflowState) -> StepResult:
    rows = []
    if state.run_log_path and state.run_log_path.exists() and state.gt_paths:
        run_df = pd.read_csv(state.run_log_path)
        for _, row in run_df.iterrows():
            edge_path = Path(row["edge_path"])
            gt_path = _find_gt_for_image(state, str(row["image"]))
            if not edge_path.exists() or gt_path is None or not gt_path.exists():
                continue
            pred = read_gray(edge_path)
            gt = read_gray(gt_path)
            metrics = edge_metrics_tolerance(pred, gt, tolerance=state.exp_cfg.tolerance)
            rows.append({"image": row["image"], "method": row["method"], **metrics})

    metrics_path = state.run_dir / "metrics.csv"
    summary_path = state.run_dir / "metrics_summary.csv"
    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        summary = metrics_df.groupby("method", as_index=False)[["precision_at_k", "recall_at_k", "f1_at_k", "iou_strict"]].mean()
        summary = summary.sort_values("f1_at_k", ascending=False)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    else:
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["method", "precision_at_k", "recall_at_k", "f1_at_k", "iou_strict"]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    state.metrics_path = metrics_path
    state.metrics_summary_path = summary_path
    msg = "指标评价智能体已计算 Precision@k、Recall@k、F1@k 与 IoU。" if rows else "未检测到可用 GT，指标评价智能体已生成空指标表。"
    res = StepResult(
        step=5,
        agent="指标评价智能体",
        title="Step 5 指标评价",
        status="完成",
        input="edges/、gt/、评价容差 k",
        output="metrics.csv、metrics_summary.csv",
        message=msg,
        artifacts={"metrics.csv": str(metrics_path), "metrics_summary.csv": str(summary_path)},
    )
    _log(state, res)
    _save_state(state)
    return res


def step6_result_analyst_agent(state: WorkflowState) -> StepResult:
    metrics_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    if state.metrics_path and state.metrics_path.exists():
        metrics_df = pd.read_csv(state.metrics_path)
    if state.metrics_summary_path and state.metrics_summary_path.exists():
        summary_df = pd.read_csv(state.metrics_summary_path)

    lines = [
        "# Step 6 结果分析智能体输出：实验现象解释",
        "",
        "## 分析依据",
        "结果分析智能体基于指标评价智能体生成的 metrics.csv、metrics_summary.csv，以及各算法边缘图文件进行解释。",
        "",
    ]
    analysis = []
    if not summary_df.empty:
        best = summary_df.iloc[0]
        lines.extend([
            "## 指标层面结论",
            f"按平均 F1@k 排序，当前表现最好的方法是 **{best['method']}**，其 F1@k 为 {best['f1_at_k']:.3f}。",
            "",
            "## 方法现象解释",
        ])
        for _, row in summary_df.iterrows():
            method = row["method"]
            p, r, f1 = row["precision_at_k"], row["recall_at_k"], row["f1_at_k"]
            if method == "Canny":
                note = "Canny 通常边缘较细，双阈值连接有助于抑制孤立噪声，但阈值偏高时可能漏检弱裂缝。"
            elif method in {"Sobel", "Prewitt"}:
                note = "一阶梯度算子计算简单，但对水泥纹理、光照变化和细小噪声较敏感，容易产生背景边缘。"
            elif method == "LoG":
                note = "LoG 对灰度二阶变化敏感，能够响应细节，但在纹理复杂区域容易产生断裂或虚假边缘。"
            elif method == "Morphological Gradient":
                note = "形态学梯度能突出结构边界，但边缘可能偏粗，对结构元素大小较敏感。"
            else:
                note = "该方法的表现需要结合边缘图进一步观察。"
            line = f"- **{method}**：Precision@k={p:.3f}, Recall@k={r:.3f}, F1@k={f1:.3f}。{note}"
            lines.append(line)
            analysis.append({"method": method, "precision_at_k": float(p), "recall_at_k": float(r), "f1_at_k": float(f1), "note": note})
    else:
        lines.extend([
            "## 指标层面结论",
            "当前没有可用的 GT 或指标表为空，因此无法进行定量优劣判断。系统仍保存了边缘图，可用于视觉对比。",
        ])

    lines.extend([
        "",
        "## 多智能体协作意义",
        "结果分析智能体不是独立运行，而是接收实验执行智能体输出的边缘图和指标评价智能体输出的指标表，再对实验现象进行解释。这体现了阶段产物在智能体之间的传递。",
    ])

    report_path = state.run_dir / "analysis_report.md"
    json_path = state.run_dir / "agent_analysis.json"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    state.analysis_report_path = report_path
    state.agent_analysis_path = json_path
    res = StepResult(
        step=6,
        agent="结果分析智能体",
        title="Step 6 结果分析",
        status="完成",
        input="metrics.csv、metrics_summary.csv、edges/、algorithm_plan.md",
        output="analysis_report.md、agent_analysis.json",
        message="结果分析智能体已结合指标和方法特点生成实验现象解释。",
        artifacts={"analysis_report.md": str(report_path), "agent_analysis.json": str(json_path)},
    )
    _log(state, res)
    _save_state(state)
    return res


def step7_report_exporter(state: WorkflowState) -> StepResult:
    text = build_research_report(
        task=state.task,
        run_dir=state.run_dir,
        selected_algorithms=state.selected_algorithms,
        literature_summary_path=state.literature_summary_path,
        algorithm_plan_path=state.algorithm_plan_md_path,
        metrics_path=state.metrics_path,
        metrics_summary_path=state.metrics_summary_path,
        analysis_report_path=state.analysis_report_path,
        agent_logs=state.agent_logs,
    )
    report_path = save_report(state.run_dir / "research_report.md", text)
    zip_path = zip_directory(state.run_dir, state.run_dir / "experiment_package.zip")
    state.final_report_path = report_path
    state.zip_path = zip_path
    res = StepResult(
        step=7,
        agent="报告导出模块",
        title="Step 7 报告导出",
        status="完成",
        input="前六个智能体全部阶段产物",
        output="research_report.md、experiment_package.zip",
        message="已整合文献、数据、算法、实验、指标、分析和协作日志，生成最终报告与实验压缩包。",
        artifacts={"research_report.md": str(report_path), "experiment_package.zip": str(zip_path)},
    )
    _log(state, res)
    _save_state(state)
    return res


def run_workflow_iter(cfg: WorkflowConfig) -> Generator[tuple[WorkflowState, StepResult], None, WorkflowState]:
    state = init_state(cfg)
    steps = [
        lambda: step1_literature_agent(state),
        lambda: step2_data_manager_agent(state, cfg),
        lambda: step3_algorithm_designer_agent(state),
        lambda: step4_experiment_executor_agent(state),
        lambda: step5_metric_evaluator_agent(state),
        lambda: step6_result_analyst_agent(state),
    ]
    for step in steps:
        res = step()
        yield state, res
    if cfg.export_report:
        res = step7_report_exporter(state)
        yield state, res
    return state


def run_full_workflow(cfg: WorkflowConfig) -> WorkflowState:
    state = None
    for state, _ in run_workflow_iter(cfg):
        pass
    if state is None:
        state = init_state(cfg)
    return state
