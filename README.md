# Edge-MAS：基于 AutoGen 的一键式多智能体边缘检测科研协作系统

本项目是一个面向边缘检测实验流程的多智能体协作科研自助系统。新版将 **AutoGen 设为多智能体协作核心依赖**：用户点击“一键启动多智能体协作流程”后，系统必须先通过 AutoGen 启动总控规划智能体和六个专业智能体，生成协作计划；AutoGen 成功返回后，Python 工具层才继续执行文献保存、图像处理、算法运行、指标计算和结果分析。

系统使用的边缘检测算法：Sobel、Prewitt、LoG、Canny、Morphological Gradient。

## 1. 智能体构成

- 总控规划智能体：理解需求、分解任务、调度其他智能体。
- 文献调研智能体：整理边缘检测相关文献并保存文献记录。
- 数据管理智能体：读取图像、mask，完成灰度化、配对和 edge GT 生成。
- 算法设计智能体：生成公开边缘检测算法对比方案。
- 实验执行智能体：批量运行算法并保存边缘结果。
- 指标评价智能体：计算 Precision@k、Recall@k、F1@k、IoU。
- 结果分析智能体：解释噪声、断裂、边缘粗细、漏检等实验现象。

## 2. 一键协作流程

用户只需输入研究任务、选择数据来源和算法，然后点击：

```text
🚀 启动多智能体协作流程
```

系统自动完成：

```text
AutoGen 总控规划
↓
Step 1 文献调研智能体
↓
Step 2 数据管理智能体
↓
Step 3 算法设计智能体
↓
Step 4 实验执行智能体
↓
Step 5 指标评价智能体
↓
Step 6 结果分析智能体
↓
Step 7 报告导出（可选）
```

## 3. 主要输出文件

- `autogen_conversation.json`、`autogen_conversation.md`：AutoGen 多智能体协作记录。
- `literature_records.csv`、`literature_summary.md`：文献调研智能体输出。
- `processed_images/`、`gt/`、`file_pairs.csv`：数据管理智能体输出。
- `algorithm_plan.json`、`algorithm_plan.md`：算法设计智能体输出。
- `edges/`、`experiment_config.json`、`run_log.csv`：实验执行智能体输出。
- `metrics.csv`、`metrics_summary.csv`：指标评价智能体输出。
- `analysis_report.md`、`agent_analysis.json`：结果分析智能体输出。
- `research_report.md`、`experiment_package.zip`：报告导出模块输出。

## 4. 运行环境

建议使用anaconda创建虚拟环境。AutoGen 与本地模型服务为必要条件。

本项目默认使用 Ollama 的 OpenAI-compatible 接口：

```text
模型：qwen2.5:3b
base_url：http://localhost:11434/v1
api_key：ollama
```

## 5. 安装与运行

### 5.1 创建虚拟环境

```bash
conda create -n autogenstudio python=3.11 –y
conda activate autogenstudio
```

### 5.2 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5.3 启动 Ollama 与模型

```bash
ollama pull qwen2.5:3b
ollama serve
```

如果 Ollama 已经在后台运行，`ollama serve` 可能提示端口已占用，这通常不是错误。

### 5.4 启动系统

```bash
命令行输入
streamlit run app.py
```
会自动到前端系统界面
浏览器访问：

```text
http://localhost:8501
```

## 6. AutoGen 命名规则

AutoGen / OpenAI-compatible message 的 agent name 只能使用英文、数字、下划线或短横线。因此系统内部使用英文安全名：

```text
coordinator_agent         → 总控规划智能体
literature_agent          → 文献调研智能体
data_manager_agent        → 数据管理智能体
algorithm_designer_agent  → 算法设计智能体
experiment_executor_agent → 实验执行智能体
metric_evaluator_agent    → 指标评价智能体
result_analyst_agent      → 结果分析智能体
```

前端和报告中仍显示中文智能体名称。
