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