# Edge-MAS 多智能体边缘检测实验报告

## 1. 研究任务

比较 Sobel、Prewitt、LoG、Canny 和形态学梯度在裂缝图像边缘检测中的效果，并分析不同算法的噪声、断裂和边缘粗细差异。

## 2. 多智能体协作流程

本系统将边缘检测实验流程拆分为六个自动执行阶段：文献调研、数据管理、算法设计、实验执行、指标评价和结果分析。各阶段由不同智能体负责，并通过共享文件与工作流状态传递阶段性产物。

- **Step 0 总控规划智能体**：输入：用户研究任务、数据来源、算法选择、实验参数、AutoGen 模型配置；输出：autogen_conversation.json、autogen_conversation.md、六阶段多智能体协作计划；说明：AutoGen 已成功启动总控规划智能体和六个专业智能体，完成协作任务分解；系统随后按照 AutoGen 协作计划进入工具执行阶段。
- **Step 1 文献调研智能体**：输入：研究任务、算法关键词、边缘检测主题词；输出：literature_records.csv、literature_records.jsonl、literature_summary.md；说明：文献调研智能体已保存边缘检测种子文献，并生成文献调研摘要，供算法设计智能体参考。
- **Step 2 数据管理智能体**：输入：原始图像、mask 图、数据来源设置；输出：processed_images/、gt/、file_pairs.csv；说明：数据管理智能体已完成 3 张图像的灰度化和文件配对；生成 3 个 edge GT。
- **Step 3 算法设计智能体**：输入：研究任务、文献调研摘要、数据情况、用户选择算法；输出：algorithm_plan.json、algorithm_plan.md；说明：算法设计智能体已生成公开边缘检测算法对比方案，并明确公平对比原则。
- **Step 4 实验执行智能体**：输入：algorithm_plan.json、processed_images/、实验参数；输出：edges/、experiment_config.json、run_log.csv；说明：实验执行智能体已运行 4 种算法，生成 12 张边缘检测结果图。
- **Step 5 指标评价智能体**：输入：edges/、gt/、评价容差 k；输出：metrics.csv、metrics_summary.csv；说明：指标评价智能体已计算 Precision@k、Recall@k、F1@k 与 IoU。
- **Step 6 结果分析智能体**：输入：metrics.csv、metrics_summary.csv、edges/、algorithm_plan.md；输出：analysis_report.md、agent_analysis.json；说明：结果分析智能体已结合指标和方法特点生成实验现象解释。

## 3. 文献调研智能体产物

# Step 1 文献调研智能体输出：文献调研摘要

## 研究任务
比较 Sobel、Prewitt、LoG、Canny 和形态学梯度在裂缝图像边缘检测中的效果，并分析不同算法的噪声、断裂和边缘粗细差异。

## 本次实验涉及的公开边缘检测方法
Sobel、Prewitt、LoG、Canny

## 文献调研结论
文献调研智能体围绕传统边缘检测算子、多阶段 Canny 框架、形态学梯度、深度学习边缘检测以及边界评价指标整理了种子文献。该产物用于支撑后续算法设计智能体生成算法对比方案。

## 已保存文献记录
1. **A Computational Approach to Edge Detection** (1986) — 经典 Canny 边缘检测框架，包括平滑、梯度、非极大值抑制和双阈值连接思想。
2. **Gradient-based edge operators** (1960s-1970s) — 一阶梯度边缘检测思想，计算简单，常作为传统边缘检测基线。
3. **Theory of Edge Detection** (1980) — 基于高斯平滑和二阶导数零交叉的边缘检测思想，是 LoG 方法的重要理论来源。
4. **Morphological image analysis foundations** (1980s) — 形态学梯度利用膨胀与腐蚀差分提取结构边界，适合展示非微分型边缘检测思想。
5. **Holistically-Nested Edge Detection** (2015) — 深度学习边缘检测代表性工作之一，用于说明边缘检测从手工算子到学习型方法的发展。
6. **Boundary detection evaluation with tolerance** (2000s) — 边缘位置可能存在轻微偏移，因此常采用带容差的 Precision、Recall 和 F-measure。

## 4. 算法设计智能体产物

# Step 3 算法设计智能体输出：算法对比方案

## 研究任务
比较 Sobel、Prewitt、LoG、Canny 和形态学梯度在裂缝图像边缘检测中的效果，并分析不同算法的噪声、断裂和边缘粗细差异。

## 对比算法
- **Sobel**：一阶梯度边缘算子，计算简单，适合作为传统梯度基线。
- **Prewitt**：一阶差分边缘算子，与 Sobel 类似但卷积权重更均匀。
- **LoG**：高斯平滑后使用拉普拉斯二阶导数寻找边缘，对细节和噪声较敏感。
- **Canny**：多阶段边缘检测方法，包含平滑、梯度、非极大值抑制和双阈值连接。

## 公平对比原则
- 所有算法使用同一批灰度图像作为输入。
- 所有算法结果保存为二值边缘图。
- 有 mask 时统一使用同一 edge GT 和相同容差 k 计算指标。
- 实验参数保存到 experiment_config.json，保证可复现。

## 5. 实验设置

- 实验目录：`D:\xiaodingdang\Users\Desktop\edge_autogen\data\results\run_20260622_195721`
- 对比算法：Sobel, Prewitt, LoG, Canny
- 边缘检测任务类型：裂缝图像边缘提取与算法对比

## 6. 指标评价结果

### 6.1 方法平均指标

| method   |   precision_at_k |   recall_at_k |   f1_at_k |   iou_strict |
|:---------|-----------------:|--------------:|----------:|-------------:|
| Canny    |         0.656493 |      0.622554 |  0.638771 |   0          |
| Prewitt  |         0.161396 |      1        |  0.277238 |   0.0321502  |
| Sobel    |         0.158313 |      1        |  0.27264  |   0.0305764  |
| LoG      |         0.162777 |      0.848361 |  0.27204  |   0.00983373 |

### 6.2 单图像指标明细

| image                  | method   |   precision_at_k |   recall_at_k |   f1_at_k |   iou_strict |   pred_edge_pixels |   gt_edge_pixels |
|:-----------------------|:---------|-----------------:|--------------:|----------:|-------------:|-------------------:|-----------------:|
| demo_crack_01_gray.png | Sobel    |         0.190774 |      1        |  0.32042  |   0.0581439  |              18729 |             1144 |
| demo_crack_01_gray.png | Prewitt  |         0.193643 |      1        |  0.324457 |   0.0588486  |              18720 |             1144 |
| demo_crack_01_gray.png | LoG      |         0.226255 |      0.967657 |  0.366756 |   0.00930469 |               4822 |             1144 |
| demo_crack_01_gray.png | Canny    |         0.975149 |      0.951923 |  0.963396 |   0          |               1006 |             1144 |
| demo_crack_02_gray.png | Sobel    |         0.148936 |      1        |  0.259259 |   0.0192199  |              18753 |             1133 |
| demo_crack_02_gray.png | Prewitt  |         0.152137 |      1        |  0.264095 |   0.0215076  |              18720 |             1133 |
| demo_crack_02_gray.png | LoG      |         0.140684 |      0.811121 |  0.23978  |   0.0090535  |               4997 |             1133 |
| demo_crack_02_gray.png | Canny    |         0.577491 |      0.552515 |  0.564727 |   0          |               1084 |             1133 |
| demo_crack_03_gray.png | Sobel    |         0.13523  |      1        |  0.238242 |   0.0143655  |              18879 |             1104 |
| demo_crack_03_gray.png | Prewitt  |         0.138408 |      1        |  0.243161 |   0.0160943  |              18720 |             1104 |
| demo_crack_03_gray.png | LoG      |         0.121392 |      0.766304 |  0.209583 |   0.011143   |               4885 |             1104 |
| demo_crack_03_gray.png | Canny    |         0.41684  |      0.363225 |  0.38819  |   0          |                962 |             1104 |

## 7. 结果分析智能体产物

# Step 6 结果分析智能体输出：实验现象解释

## 分析依据
结果分析智能体基于指标评价智能体生成的 metrics.csv、metrics_summary.csv，以及各算法边缘图文件进行解释。

## 指标层面结论
按平均 F1@k 排序，当前表现最好的方法是 **Canny**，其 F1@k 为 0.639。

## 方法现象解释
- **Canny**：Precision@k=0.656, Recall@k=0.623, F1@k=0.639。Canny 通常边缘较细，双阈值连接有助于抑制孤立噪声，但阈值偏高时可能漏检弱裂缝。
- **Prewitt**：Precision@k=0.161, Recall@k=1.000, F1@k=0.277。一阶梯度算子计算简单，但对水泥纹理、光照变化和细小噪声较敏感，容易产生背景边缘。
- **Sobel**：Precision@k=0.158, Recall@k=1.000, F1@k=0.273。一阶梯度算子计算简单，但对水泥纹理、光照变化和细小噪声较敏感，容易产生背景边缘。
- **LoG**：Precision@k=0.163, Recall@k=0.848, F1@k=0.272。LoG 对灰度二阶变化敏感，能够响应细节，但在纹理复杂区域容易产生断裂或虚假边缘。

## 多智能体协作意义
结果分析智能体不是独立运行，而是接收实验执行智能体输出的边缘图和指标评价智能体输出的指标表，再对实验现象进行解释。这体现了阶段产物在智能体之间的传递。

## 8. 结论

本系统以多智能体协作为核心，将边缘检测科研任务从单一算法调用扩展为完整实验流程。用户输入研究任务和数据后，系统自动完成文献整理、数据预处理、算法方案生成、批量实验执行、定量指标评价和结果现象解释。该流程能够清晰体现不同智能体之间的分工、产物传递和顺序协作关系。
