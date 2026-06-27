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

用户只需在系统左侧输入研究任务、选择数据来源和算法，然后点击：

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
模型：qwen2.5:3b-instruct-q4_0
base_url：http://localhost:11434/v1
api_key：ollama
```

## 5. 安装与运行

### 5.1 创建虚拟环境

```bash
conda create -n autogenstudio python=3.11 –y
conda activate autogenstudio
```

### 5.2 在命令行界面进入项目目录下安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5.3 启动 Ollama 与模型

```bash
ollama pull qwen2.5:3b-instruct-q4_0
ollama serve
```

如果 Ollama 已经在后台运行，`ollama serve` 可能提示端口已占用，这通常不是错误。

### 5.4 启动系统

```bash
命令行输入
streamlit run app.py
```
会自动到前端系统界面


```text
也可以在浏览器访问：
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

## 7. 系统界面展示
### 7.1 一键启动系统
<img width="1845" height="851" alt="image" src="https://github.com/user-attachments/assets/f6176d58-6bec-4ba4-b902-13e8f0a9c541" />


autogen启动需要等待一会，（一定要提前打开ollama并确定本地模型已拉取，并且和网站左侧填写的模型是一样的，如果本地不是和我的qwen2.5:3b-instruct-q4_0模型一样的话，也可以使用自己已有的模型，把自己的模型填写到系统前端的左侧栏中）

<img width="1400" height="750" alt="image" src="https://github.com/user-attachments/assets/ef888b64-8229-4ed4-9ef7-ec604a261433" />


成功运行后界面是这样子的：
<img width="1260" height="729" alt="image" src="https://github.com/user-attachments/assets/bf2e6c91-9fa7-4bb7-885e-30427412b781" />

### 7.2 点击查看流程的全部产物

<img width="1366" height="784" alt="image" src="https://github.com/user-attachments/assets/855e5062-1391-403c-bdbd-e1b1b1c74a66" />


对于每一个agent都有的说明记录：

<img width="1365" height="733" alt="image" src="https://github.com/user-attachments/assets/a7c54753-ad1f-41ae-aa37-a7bf8dde5984" />

### 7.3 点击查看文献库

agent自己拉取的文献：
<img width="1403" height="694" alt="image" src="https://github.com/user-attachments/assets/2acf16eb-66e0-4fae-9725-afd52da76b05" />

### 7.4 点击查看实验结果，也就是图片处理效果
说明了处理的是哪里的图片和处理效果一览
<img width="1401" height="718" alt="image" src="https://github.com/user-attachments/assets/2032aeb7-324d-4223-acc1-c3b858ecf57b" />

### 7.5 点击查看指标分析
选出了最佳方法，最佳 F1@k，评价容差 ，并计算出了方法平均指标
<img width="1398" height="719" alt="image" src="https://github.com/user-attachments/assets/0cda23c8-3245-4e4c-b690-d3ae758b2c6a" />

图标展示更直观：
<img width="1400" height="701" alt="image" src="https://github.com/user-attachments/assets/60e708be-1df1-42e8-b6e1-552068bd3d47" />

单个图像的指标明细也用表直观列出：
<img width="1395" height="611" alt="image" src="https://github.com/user-attachments/assets/1d497184-44c3-48d3-b140-338d8a803aa8" />

最后分析和解释实验现象：
<img width="1423" height="784" alt="image" src="https://github.com/user-attachments/assets/702b07c6-ceca-412a-b459-75cdd29f3cda" />

### 7.6 点击查看指标分析

可以下载全流程的实验报告也可在网站在线查看
<img width="1415" height="720" alt="image" src="https://github.com/user-attachments/assets/d33d2835-8a1c-4ab7-89cc-611ce3a7ff27" />











