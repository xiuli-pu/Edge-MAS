from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from edge_mas.agents.autogen_runner import run_autogen_sync
from edge_mas.agents.registry import AGENTS, AUTO_GEN_NAME_MAP, COORDINATOR
from edge_mas.config import ExperimentConfig, LLMConfig, RESULTS_DIR, UPLOADS_DIR
from edge_mas.literature.manager import LiteratureRecord, append_manual_record, load_literature_records, save_seed_literature, save_uploaded_literature
from edge_mas.report.writer import build_research_report, save_report
from edge_mas.utils import save_uploaded_files, timestamp, zip_directory
from edge_mas.workflow.pipeline import WorkflowConfig, WorkflowState, run_workflow_iter, step7_report_exporter

st.set_page_config(page_title="Edge-MAS AutoGen 多智能体协作系统", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 2rem; overflow: visible;}
.main-title {
    font-size: clamp(1.65rem, 2.4vw, 2.1rem);
    font-weight: 800;
    line-height: 1.35;
    padding-top: 0.25rem;
    padding-bottom: 0.25rem;
    margin-top: 0.15rem;
    margin-bottom: 0.35rem;
    overflow: visible;
    white-space: normal;
    word-break: keep-all;
    font-family: "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif;
}
.sub-title {font-size: 1.0rem; line-height: 1.75; color: #666; margin-top: 0.1rem; margin-bottom: 1.0rem; font-family: "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif;}
.step-card {border: 1px solid #e7e7e7; border-radius: 14px; padding: 14px 16px; margin-bottom: 10px; background: #ffffff; box-shadow: 0 1px 4px rgba(0,0,0,0.04);} 
.step-card-done {border-left: 6px solid #2ca02c;}
.step-card-wait {border-left: 6px solid #bbb;}
.step-title {font-weight: 800; font-size: 1.02rem; line-height: 1.55;}
.step-meta {color: #555; font-size: 0.92rem; line-height: 1.55;}
.output-badge {background: #f1f5ff; color: #24428f; padding: 2px 8px; border-radius: 999px; font-size: 0.82rem;}
.agent-badge {background: #fff3cd; color: #7a5b00; padding: 2px 8px; border-radius: 999px; font-size: 0.82rem;}
.autogen-badge {background: #eaf7ff; color: #005c83; padding: 2px 8px; border-radius: 999px; font-size: 0.82rem;}
.big-button-note {background:#f7f9fc; border-radius:12px; padding:14px 16px; border:1px solid #e8edf5; line-height:1.75;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Edge-MAS：基于 AutoGen 的一键式多智能体边缘检测科研协作系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">用户只需输入任务、选择数据并点击启动，系统首先强制启动 AutoGen 多智能体协作，再按 Step 1 → Step 6 执行工具流程；Step 7 报告导出可选。</div>', unsafe_allow_html=True)

if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = None
if "step_results" not in st.session_state:
    st.session_state.step_results = []
if "uploaded_image_paths" not in st.session_state:
    st.session_state.uploaded_image_paths = []
if "uploaded_mask_paths" not in st.session_state:
    st.session_state.uploaded_mask_paths = []
if "autogen_rows" not in st.session_state:
    st.session_state.autogen_rows = []

ALL_ALGORITHMS = ["Sobel", "Prewitt", "LoG", "Canny", "Morphological Gradient"]

def render_step_card(step: int, agent: str, title: str, output: str, status: str = "等待", message: str = "") -> None:
    cls = "step-card step-card-done" if status == "完成" else "step-card step-card-wait"
    icon = "✅" if status == "完成" else "⬜"
    st.markdown(
        f"""
<div class="{cls}">
  <div class="step-title">{icon} Step {step}｜{title}</div>
  <div class="step-meta"><span class="agent-badge">{agent}</span> <span class="output-badge">输出：{output}</span></div>
  <div class="step-meta">{message}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def latest_state() -> WorkflowState | None:
    return st.session_state.workflow_state


def read_csv_safe(path: Path | None) -> pd.DataFrame:
    if path and Path(path).exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

with st.sidebar:
    st.header("一键流程配置")
    task = st.text_area(
        "研究任务",
        value="比较 Sobel、Prewitt、LoG、Canny 和形态学梯度在裂缝图像边缘检测中的效果，并分析不同算法的噪声、断裂和边缘粗细差异。",
        height=130,
    )
    selected_algorithms = st.multiselect("公开边缘检测算法", ALL_ALGORITHMS, default=["Sobel", "Prewitt", "LoG", "Canny"])
    data_source = st.radio("数据来源", ["使用系统演示数据", "上传自己的图像和 mask"], index=0)
    export_report = st.checkbox("Step 7：自动生成 research_report.md", value=True)

    st.divider()
    st.header("实验参数")
    tolerance = st.slider("指标评价容差 k 像素", 0, 5, 2, 1)
    sigma = st.slider("平滑 sigma", 0.2, 4.0, 1.2, 0.1)
    canny_low = st.slider("Canny 低阈值", 10, 200, 60, 5)
    canny_high = st.slider("Canny 高阈值", 30, 300, 160, 5)
    log_sigma = st.slider("LoG sigma", 0.2, 4.0, 1.2, 0.1)
    morph_kernel = st.slider("形态学核大小", 3, 9, 3, 2)

    st.divider()
    st.header("AutoGen 必选配置")
    st.caption("本项目把 AutoGen 作为多智能体协作的核心依赖；未安装 AutoGen 或模型服务未启动时，流程不会继续执行。")
    model = st.text_input("模型", value="qwen2.5:3b-instruct-q4_0")
    base_url = st.text_input("base_url", value="http://localhost:11434/v1")
    api_key = st.text_input("api_key", value="ollama", type="password")

exp_cfg = ExperimentConfig(
    tolerance=tolerance,
    sigma=sigma,
    canny_low=canny_low,
    canny_high=canny_high,
    log_sigma=log_sigma,
    morph_kernel=morph_kernel,
)
llm_cfg = LLMConfig(model=model, base_url=base_url, api_key=api_key)

tab_workbench, tab_outputs, tab_lit, tab_results, tab_analysis, tab_report, tab_help = st.tabs([
    "🚀 一键协作工作台",
    "🧩 流程产物",
    "📚 文献库详情",
    "🖼️ 实验结果详情",
    "📊 指标与分析",
    "📝 报告导出",
    "⚙️ 使用说明",
])

with tab_workbench:
    st.subheader("一键启动：Step 1 → Step 6 自动协作")
    st.markdown(
        """
<div class="big-button-note">
<strong>操作逻辑：</strong>先在左侧设置研究任务、算法、数据来源和 AutoGen 模型参数，然后点击下面的按钮。系统会先调用 AutoGen 的总控规划智能体与六个专业智能体生成协作计划；AutoGen 成功后，再按顺序执行工具流程：文献调研 → 数据管理 → 算法设计 → 实验执行 → 指标评价 → 结果分析。第七步报告导出由左侧复选框控制。
</div>
""",
        unsafe_allow_html=True,
    )

    if data_source == "上传自己的图像和 mask":
        c1, c2 = st.columns(2)
        with c1:
            uploaded_images = st.file_uploader("上传原始图像，可多选", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], accept_multiple_files=True, key="pipeline_images")
        with c2:
            uploaded_masks = st.file_uploader("上传 mask，可选；最好与原图同名", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], accept_multiple_files=True, key="pipeline_masks")
        if uploaded_images:
            upload_dir = UPLOADS_DIR / f"manual_{timestamp()}"
            st.session_state.uploaded_image_paths = save_uploaded_files(uploaded_images, upload_dir / "images")
            st.session_state.uploaded_mask_paths = save_uploaded_files(uploaded_masks or [], upload_dir / "masks")
            st.success(f"已暂存 {len(st.session_state.uploaded_image_paths)} 张图像、{len(st.session_state.uploaded_mask_paths)} 张 mask。")

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        start = st.button("🚀 启动多智能体协作流程", type="primary", use_container_width=True)
    with col_b:
        st.metric("自动执行阶段", "6 + 可选报告")
    with col_c:
        st.metric("AutoGen 智能体", "6 个 + 总控")

    st.markdown("### 协作流水线")
    existing = {r["step"]: r for r in st.session_state.step_results} if st.session_state.step_results else {}
    for spec in AGENTS:
        if spec.step in existing:
            r = existing[spec.step]
            render_step_card(spec.step, r["agent"], r["title"], r["output"], "完成", r["message"])
        else:
            render_step_card(spec.step, spec.display_name, spec.responsibility, spec.output, "等待", "等待总控规划智能体调度。")
    if export_report:
        render_step_card(7, "报告导出模块", "报告导出", "research_report.md、experiment_package.zip", "完成" if 7 in existing else "等待", existing.get(7, {}).get("message", "前六步完成后自动整合全部产物。"))

    if start:
        if not selected_algorithms:
            st.error("请至少选择一种算法。")
        elif data_source == "上传自己的图像和 mask" and not st.session_state.uploaded_image_paths:
            st.error("你选择了上传数据，但还没有上传原始图像。")
        else:
            st.session_state.step_results = []
            st.session_state.workflow_state = None
            st.session_state.autogen_rows = []
            use_demo = data_source == "使用系统演示数据"
            wf_cfg = WorkflowConfig(
                task=task,
                selected_algorithms=selected_algorithms,
                exp_cfg=exp_cfg,
                use_demo_data=use_demo,
                image_paths=st.session_state.uploaded_image_paths,
                mask_paths=st.session_state.uploaded_mask_paths,
                export_report=export_report,
                demo_count=3,
                llm_cfg=llm_cfg,
            )
            progress = st.progress(0)
            status = st.status("正在启动 AutoGen 多智能体协作。AutoGen 成功后才会进入工具执行流程……", expanded=True)
            total = 7 if export_report else 6
            final_state = None
            try:
                with status:
                    for state, result in run_workflow_iter(wf_cfg):
                        final_state = state
                        st.session_state.step_results.append(result.__dict__)
                        st.write(f"✅ {result.title}：{result.message}")
                        if state.autogen_rows:
                            st.session_state.autogen_rows = state.autogen_rows
                        progress.progress(min(result.step / total, 1.0))
                    st.write("🎉 基于 AutoGen 的一键多智能体协作流程完成。")
                st.session_state.workflow_state = final_state
                st.rerun()
            except Exception as exc:
                st.error(f"AutoGen 多智能体协作启动失败，流程已停止。原因：{exc}")
                st.info("请确认：1）已安装 requirements.txt 中的 AutoGen 依赖；2）Ollama 已启动；3）模型名称与 base_url 正确。")

with tab_outputs:
    st.subheader("流程产物总览")
    state = latest_state()
    if state is None:
        st.info("还没有运行一键协作流程。请先到“🚀 一键协作工作台”点击启动。")
    else:
        st.success(f"当前实验目录：{state.run_dir}")
        cols = st.columns(3)
        cols[0].metric("处理图像", len(state.processed_image_paths))
        cols[1].metric("Edge GT", len(state.gt_paths))
        cols[2].metric("算法数量", len(state.selected_algorithms))
        st.markdown("### 智能体阶段产物")
        for log in state.agent_logs:
            if int(log.get("step", -1)) == 0:
                continue
            with st.expander(f"Step {log.get('step')}｜{log.get('agent')}｜{log.get('title')}", expanded=True):
                st.write(f"**输入：** {log.get('input')}")
                st.write(f"**输出：** {log.get('output')}")
                st.write(f"**说明：** {log.get('message')}")
                artifacts = log.get("artifacts", {})
                if artifacts:
                    st.json(artifacts)
        if st.session_state.autogen_rows:
            st.markdown("### AutoGen 多智能体协作记录")
            for row in st.session_state.autogen_rows:
                with st.expander(row.get("agent", "智能体")):
                    st.write(row.get("content", ""))

with tab_lit:
    st.subheader("文献库详情")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("保存/更新边缘检测种子文献", use_container_width=True):
            csv_path, jsonl_path = save_seed_literature()
            st.success(f"已保存：{csv_path} 和 {jsonl_path}")
    with c2:
        lit_files = st.file_uploader("上传自己的文献 PDF/TXT/BibTeX", type=["pdf", "txt", "md", "bib"], accept_multiple_files=True)
        if lit_files and st.button("保存上传文献", use_container_width=True):
            saved = save_uploaded_literature(lit_files)
            st.success(f"已保存 {len(saved)} 个文献文件。")
    with st.expander("手动添加一条文献记录"):
        topic = st.text_input("主题", value="Canny")
        title = st.text_input("题名", value="")
        authors = st.text_input("作者", value="")
        year = st.text_input("年份", value="")
        source = st.text_input("来源", value="")
        note = st.text_area("说明", value="")
        keywords = st.text_input("关键词", value="edge detection")
        if st.button("添加到文献库"):
            append_manual_record(LiteratureRecord(topic, title, authors, year, source, note, keywords))
            st.success("已添加。")
    records = load_literature_records()
    st.dataframe(pd.DataFrame(records), use_container_width=True)

with tab_results:
    st.subheader("实验结果详情")
    state = latest_state()
    if state is None:
        st.info("请先运行一键协作流程。")
    else:
        pair_df = read_csv_safe(state.file_pairs_path)
        if not pair_df.empty:
            st.markdown("### 文件配对表")
            st.dataframe(pair_df, use_container_width=True)
        st.markdown("### 边缘检测结果预览")
        edge_files = []
        if state.edges_dir and Path(state.edges_dir).exists():
            edge_files = list(Path(state.edges_dir).rglob("*.png"))[:16]
        if edge_files:
            cols = st.columns(4)
            for i, p in enumerate(edge_files):
                cols[i % 4].image(str(p), caption=f"{p.parent.name} / {p.name}", use_container_width=True)
        else:
            st.info("没有找到边缘结果图。")

with tab_analysis:
    st.subheader("指标评价与结果分析")
    state = latest_state()
    if state is None:
        st.info("请先运行一键协作流程。")
    else:
        metrics_df = read_csv_safe(state.metrics_path)
        summary_df = read_csv_safe(state.metrics_summary_path)
        c1, c2, c3 = st.columns(3)
        if not summary_df.empty:
            best = summary_df.iloc[0]
            c1.metric("最佳方法", str(best["method"]))
            c2.metric("最佳 F1@k", f"{best['f1_at_k']:.3f}")
            c3.metric("评价容差 k", state.exp_cfg.tolerance)
            st.markdown("### 方法平均指标")
            st.dataframe(summary_df, use_container_width=True)
            chart_df = summary_df.set_index("method")[["precision_at_k", "recall_at_k", "f1_at_k", "iou_strict"]]
            st.bar_chart(chart_df)
        else:
            st.warning("没有可用指标。通常是因为没有上传 mask 或演示数据未成功生成 GT。")
        if not metrics_df.empty:
            st.markdown("### 单图像指标明细")
            st.dataframe(metrics_df, use_container_width=True)
        if state.analysis_report_path and Path(state.analysis_report_path).exists():
            st.markdown("### 结果分析智能体报告")
            st.markdown(Path(state.analysis_report_path).read_text(encoding="utf-8"))

with tab_report:
    st.subheader("报告导出")
    state = latest_state()
    if state is None:
        st.info("请先运行一键协作流程。")
    else:
        if st.button("生成/重新生成 research_report.md", type="primary"):
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
            state.final_report_path = save_report(Path(state.run_dir) / "research_report.md", text)
            state.zip_path = zip_directory(Path(state.run_dir), Path(state.run_dir) / "experiment_package.zip")
            st.session_state.workflow_state = state
            st.success("报告与压缩包已生成。")
        if state.final_report_path and Path(state.final_report_path).exists():
            report_text = Path(state.final_report_path).read_text(encoding="utf-8")
            st.download_button("下载 Markdown 报告", report_text.encode("utf-8"), file_name="research_report.md", mime="text/markdown")
            st.markdown(report_text)
        if state.zip_path and Path(state.zip_path).exists():
            st.download_button("下载完整实验结果 ZIP", Path(state.zip_path).read_bytes(), file_name="experiment_package.zip", mime="application/zip")

with tab_help:
    st.subheader("使用说明：为什么这才体现多智能体协作？")
    st.markdown(
        """
### 正确使用顺序

1. 在左侧输入研究任务；
2. 选择数据来源：演示数据或上传自己的图像和 mask；
3. 选择公开边缘检测算法；
4. 确认 AutoGen 模型服务已经启动，然后点击 **🚀 启动多智能体协作流程**；
5. 系统先启动 AutoGen 生成多智能体协作计划，再自动按 Step 1 到 Step 6 执行；
6. Step 7 报告导出可以自动执行，也可以后续手动执行。

### 多智能体协作体现在哪里？

- **总控规划智能体**：负责把用户任务拆成六个阶段。
- **文献调研智能体**：输出 `literature_records.csv` 和 `literature_summary.md`。
- **数据管理智能体**：输出 `processed_images/`、`gt/` 和 `file_pairs.csv`。
- **算法设计智能体**：输出 `algorithm_plan.json` 和 `algorithm_plan.md`。
- **实验执行智能体**：输出 `edges/`、`experiment_config.json` 和 `run_log.csv`。
- **指标评价智能体**：输出 `metrics.csv` 和 `metrics_summary.csv`。
- **结果分析智能体**：输出 `analysis_report.md` 和 `agent_analysis.json`。

每个智能体都生成下一个阶段要用的文件，前一个智能体的产物会被后一个智能体继承使用，这就是协作链条。

### AutoGen 核心说明

本版本的多智能体协作基于 AutoGen。系统启动时必须先调用 AutoGen 的 `RoundRobinGroupChat`，由总控规划智能体与六个专业智能体生成协作计划；只有 AutoGen 成功返回后，Python 工具层才继续执行文献保存、图像处理、算法运行、指标计算和结果分析。

系统内部 AutoGen agent 使用英文安全名，例如 `literature_agent`、`data_manager_agent`；前端展示仍使用中文名称。这可以避免 AutoGen/OpenAI-compatible 接口的 `Invalid name` 错误。
"""
    )
    st.markdown("### 内部英文名与前端中文名映射")
    st.json(AUTO_GEN_NAME_MAP)
