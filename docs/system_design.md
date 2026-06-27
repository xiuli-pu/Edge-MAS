# 系统设计说明

## 1. 设计目标

本系统以“基于 AutoGen 的多智能体协作”为中心，把边缘检测实验流程拆分为多个可执行阶段。用户不需要手动依次切换页面完成流程，而是通过一键启动，先由 AutoGen 的总控规划智能体与六个专业智能体生成协作计划，再进入 Python 工具执行阶段。

## 2. 协作链条

```text
用户研究任务
↓
AutoGen 总控规划智能体：生成六阶段任务链与协作计划
↓
文献调研智能体：保存文献记录与摘要
↓
数据管理智能体：处理图像、mask 和 edge GT
↓
算法设计智能体：生成算法对比方案
↓
实验执行智能体：批量运行边缘检测算法
↓
指标评价智能体：计算 Precision@k、Recall@k、F1@k、IoU
↓
结果分析智能体：解释噪声、断裂、粗边缘、漏检等现象
↓
报告导出模块：整合全部产物
```

## 3. 共享状态

工作流使用 `WorkflowState` 保存全部阶段产物路径。每个智能体完成后都会更新状态，并写入 `agent_logs.json` 和 `workflow_state.json`。

这意味着智能体之间不是并列展示，而是通过文件与状态传递形成依赖关系。

## 4. 公开算法

系统只使用公开算法：

- Sobel
- Prewitt
- LoG
- Canny
- Morphological Gradient

## 5. 主要输出

每次运行会生成一个 `data/results/run_YYYYMMDD_HHMMSS/` 目录，包含：

- `autogen_conversation.json`
- `autogen_conversation.md`
- `literature/`
- `processed_images/`
- `gt/`
- `edges/`
- `file_pairs.csv`
- `algorithm_plan.json`
- `algorithm_plan.md`
- `experiment_config.json`
- `run_log.csv`
- `metrics.csv`
- `metrics_summary.csv`
- `analysis_report.md`
- `agent_analysis.json`
- `research_report.md`（可选）
- `experiment_package.zip`（可选）
