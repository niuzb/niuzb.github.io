## Background and Research Motivation
Since 2019, the entire field of End-to-End Neural Diarization (EEND) research has treated speaker segmentation as a frame-level multi-label classification problem based on permutation-invariant training. Although EEND has shown great potential, recent studies have explored the combination of (local) supervised EEND segmentation and (global) unsupervised clustering. The loss function has shifted from multi-label classification (where any two speakers can be active simultaneously) to power set multi-class classification (where dedicated categories are assigned to overlapping speaker pairs). The new loss function significantly improves performance (mainly in overlapping speech) and robustness to domain mismatch, while eliminating the detection threshold hyperparameter that is crucial in the multi-label formulation.

Transforming speaker speech segmentation from "multi-label classification" to "power set multi-class classification" enables:
1. A significant improvement in **overlapping speech processing accuracy** (reduction of the main error source);
2. Enhanced **robustness to domain mismatch** (elimination of the sensitive threshold θ);
- Open-source model: https://huggingface.co/niuzb/wav2vec2-ssd
- Methodology: Verified that "short-block processing + power set classification" can solve the "class explosion" problem of traditional power set methods, providing a paradigm for subsequent research.

### What is Speaker Diarization?
It segments an audio stream into temporally continuous homogeneous segments based on "speaker identity" (e.g., "0-5s: Speaker A", "5-12s: Speaker B", "12-18s: Overlap of A and B").


#### Traditional Multi-Stage Methods (Classic Framework)
Workflow: Voice Activity Detection (VAD) -> Speaker Embedding (extracting discriminative features) -> Unsupervised Clustering (grouping by embeddings).  
Core drawbacks:
- **Error Propagation**: Errors from the previous step accumulate in subsequent steps (e.g., missed detection in VAD leads to missing embedding extraction);
- **Poor Overlapping Speech Handling**: Cannot directly detect segments with "multiple speakers talking simultaneously" and requires additional post-processing.

#### End-to-End Neural Diarization (EEND, Proposed in 2019)
Improvement Idea: Use a single neural network to directly input audio and output segmentation results, solving the error propagation problem.  
Implementation: Model the task as **frame-level multi-label classification** — each speaker corresponds to a binary classification label indicating "active or not", allowing multiple labels to be activated simultaneously (e.g., "overlap of Speaker A and B" corresponds to both A and B labels being 1). Through **Permutation-Invariant Training (PIT)**, unsupervised clustering is converted into a supervised task (Note: The core of PIT is "speaker label disorder", which requires traversing all label permutations to find the permutation with the minimum loss).  
![permutaion free loss](../assets/images/permutation%20free%20loss.png)
Limitations of EEND:
- Difficulty in predicting the "number of speakers" (performance drops sharply when the number in the test set exceeds that in training);
- Data Hunger: Each conversation is counted as only one training sample, relying on unrealistic synthetic conversations;
- Poor Scalability: The self-attention mechanism cannot handle long conversations.

#### Hybrid Methods (Best of Both Worlds)
To address the limitations of EEND, hybrid "multi-stage + end-to-end" frameworks have emerged in recent years, with the following workflow:
1. Split long conversations into short blocks (e.g., 5s);
2. Process each short block with EEND (local segmentation);
3. Concatenate short blocks using speaker embedding + unsupervised clustering (global integration).
![hybrid method](../assets/images/hybrid%20method.png)

However, the **key drawback**: Such hybrid methods still adopt the "multi-label classification" framework of EEND and fail to solve its core issues — reliance on a "detection threshold (θ)" (requires manual tuning and has a significant impact on performance) and low prediction accuracy for overlapping speech.

## From Multi-Label Classification to Power Set Multi-Class Classification
Abandon the multi-label approach of "independent binary classification for each speaker" and instead treat "all possible speaker combinations" as mutually exclusive categories, i.e., "power set multi-class classification".

### Review of Multi-Label Classification (Baseline Framework)
Taking "local maximum number of speakers Kmax = 3" as an example:
- Output Dimension: Kmax = 3 (each dimension corresponds to one speaker);
- Label Encoding: Frame-level binary vector (e.g., `[1,0,0]` indicates "only Speaker S1 is active", `[1,1,0]` indicates "overlap of S1 and S2");
- Activation Function: Sigmoid (compresses the output of each dimension to [0,1], representing the "probability of the speaker being active");
- Loss Function: Permutation-Invariant Binary Cross-Entropy (PIT-BCE) — traverse all permutations of the 3 speaker labels to find the minimum BCE loss;
- Testing Phase: A **detection threshold θ** (e.g., θ = 0.5) needs to be set; if the output probability exceeds θ, it is determined as "active" (θ requires manual tuning, is sensitive, and depends on data).

### Design of Power Set Multi-Class Classification
#### Core Idea
Treat "all possible speaker combinations (including non-speech)" as **mutually exclusive single categories** to cover all scenarios. Still taking Kmax = 3 as an example:
- Total Number of Power Set Categories: \(2^{Kmax} - 1 = 7\) (ignoring overlaps of 3 or more speakers, as they account for <2% of the data, resulting in 7 categories in total):
  1. 0: No speech;
  2. {S1}, {S2}, {S3}: Single speaker active;
  3. {S1,S2}, {S1,S3}, {S2,S3}: Two speakers overlapping.

#### Model Adaptation Modifications
Only adjustments to the output layer and loss of the neural network are required; no changes to the backbone structure are needed:

| Module                | Multi-Label Classification (Baseline) | Power Set Multi-Class Classification |
|-----------------------|---------------------------------------|--------------------------------------|
| Output Layer Dimension| Kmax = 3                              | Kpowerset = 7                        |
| Activation Function   | Sigmoid                               | Softmax (mutually exclusive between classes) |
| Loss Function         | Permutation-Invariant BCE             | Permutation-Invariant Cross-Entropy (CE) |
| Decision in Testing Phase | Relies on detection threshold θ    | Directly take argmax (no parameters) |

## 4. Experimental Results and Key Findings
### 4.1 Core Performance Comparison (DER, Lower is Better)

| Method                          | In-Domain Average DER | DIHARD III (Domain Mismatch) DER |
|---------------------------------|-----------------------|----------------------------------|
| Pre-trained Multi-Label Baseline| 23.7%                 | 25.9%                            |
| Multi-Label (Compound Training) | 25.6%                 | 33.8%                            |
| Multi-Label (Compound + Domain Adaptation) | 23.0%         | 22.4%                            |
| **Power Set (Compound Training)**| **23.5%**            | **29.9%**                        |
| **Power Set (Compound + Domain Adaptation)** | **21.7%**    | **21.3%**                        |

Key Conclusions:
1. The power set method is superior to multi-label classification **without domain adaptation** (under compound training, in-domain DER decreases from 25.6% to 23.5%, an 8% relative improvement);
2. In domain mismatch scenarios, the power set method shows a more significant improvement (DER decreases from 33.8% to 29.9%, an 11% relative improvement), due to the elimination of the sensitive threshold θ;
3. After combining power set with domain adaptation, the model achieves **SOTA** on 6 datasets including AISHELL-4 and AliMeeting.

### Why is the Power Set Better?
Through error decomposition on DIHARD III (Figure 3), it is found that the improvement mainly comes from:
- **Reduced Missed Detection of Overlapping Speech**: The overlapping missed detection rate of multi-label classification decreases from 13.1% to 9.9% for the power set (because the power set models "overlap" as an explicit category, rather than relying on threshold judgment of two labels);
- No significant changes in False Alarms and Speaker Confusion.

### Additional Value of Domain Adaptation
After fine-tuning the power set model on various DIHARD sub-domains (e.g., restaurants, clinical settings, broadcasts), it can better balance "False Alarms" and "Missed Detections" without affecting the Speaker Confusion rate. This is equivalent to using domain adaptation to replace the "manual θ tuning" in multi-label classification, further improving robustness.

## Supplementary Key Terms
- **DER (Diarization Error Rate)**: A metric for speech segmentation errors, measuring the proportion of "misclassified segment duration + missed speech duration + false positive speech duration" to the total speech duration. It is a core indicator in this field;
- **Permutation-Invariant Training (PIT)**: Since speaker labels have no fixed order (e.g., swapping S1 and S2 does not change the semantics), it is necessary to traverse all label permutations to find the minimum loss, ensuring the model does not rely on label order;
- **Power Set**: A mathematical concept referring to the set of all subsets of a given set. Here, it is used to define the categories of "all speaker combinations".

## Why Loss Functions Matter
From the experimental results, the new loss function improves performance by 20% on overlapping speech and 10% on non-overlapping speech. Referring to the evolution history of models in the face recognition field, the invention of new loss functions has consistently led to improvements in model performance.

The loss functions for face recognition models have gone through multiple development stages to continuously enhance face recognition performance. The following sorts out their evolution in chronological order:
1. **Softmax Loss**: A classic classification loss function that maximizes the prediction probability of the correct category. It is widely used in image classification. However, it only considers correct classification and does not account for inter-class distances. In face recognition tasks, it only provides separability but lacks discriminability, failing to achieve intra-class aggregation and thus unable to meet the requirements for features in face recognition.
2. **Center Loss**: Proposed in 2016, it adds a term to the Softmax Loss that pulls samples of the same class closer to the center. It sets a center for each class and minimizes intra-class distances while ensuring classification. However, it has no classification function itself and must be used in conjunction with Softmax Loss. The center is initialized with random values and is updated in real-time with the learned features. When calculating the center loss for each class, it is divided by the number of samples in that class to compute the mean, preventing unbalanced samples from causing asynchronous gradient updates across different classes. In face recognition, the empirical value of the parameter is generally set to 0.003. It has issues such as unsatisfactory intra-class distance optimization, high hardware requirements when there are many classes, difficulty in optimizing outliers with L2 norm, and applicability only to data with small differences between samples of the same class.
3. **Triplet Loss**: A triplet loss function composed of Anchor, Negative, and Positive samples. It optimizes the model by minimizing the distance between Anchor and Positive (reducing intra-class distance) and maximizing the distance between Anchor and Negative (increasing inter-class distance). Before Center Loss, it was a commonly used loss function in face recognition. However, as the number of samples increases, the number of sample pair combinations grows exponentially, resulting in severe training time consumption.
4. **L-Softmax Loss**: Adjusts Softmax Loss by converting convolution operations into vector products and changing \(cosθ\) to \(cos(mθ)\) (where \(m>1\)). This increases the decision margin, raises the learning difficulty, thereby compressing intra-class distances and expanding inter-class distances.
5. **SphereFace (A-Softmax Loss)**: Proposed in 2017, it normalizes based on L-Softmax Loss, mapping points on features to a unit hypersphere. The model's predictions depend only on the angle between \(W\) and \(X\). It proposes angular margin penalty, focusing training more on optimizing deep feature mapping and the angles of feature vectors, and reducing the problem of unbalanced sample counts. However, the calculation of the loss function requires a series of approximations, leading to unstable network training.
6. **CosFace (AM-Softmax Loss)**: Proposed in 2018, it directly adds a cosine margin penalty to the target logistic regression, using an additive cosine margin (\(cos(θ)-m\)) and normalizing feature vectors and weights. Compared with SphereFace, it has better performance, is easier to implement, and reduces the need for joint supervision with Softmax Loss.
7. **ArcFace**: Proposed in 2018, it introduces an additive angular margin loss (\(θ+m\)), also normalizing feature vectors and weights. It has a constant linear angular margin geometrically and directly optimizes radians. To ensure stable performance, it does not require joint supervision with other loss functions and performs excellently in face recognition tasks.

# 如何训练一个更好的端到端说话人语音分割模型
## 背景与研究动机
自2019以来，整个端到端神经语音分割（EEND）研究一直将说话人分割视为一个基于排列不变训练的帧级多标签分类问题。尽管EEND展现出了巨大潜力，但最近探索的（局部）有监督EEND分割与（全局）无监督聚类相结合的可能性。损失函数从多标签分类（其中任意两个说话人可以同时处于活跃状态）转向幂集多类分类（其中专门的类别被分配给重叠说话人对）。新的损失函数能能显著提升性能（主要体现在重叠语音上）和对领域失配的鲁棒性，同时消除了多标签公式中至关重要的检测阈值超参数。
将说话人语音分割从“多标签分类”转为“幂集多分类”，能：
1. 显著提升**重叠语音处理精度**（主要误差来源减少）；
2. 增强**领域不匹配鲁棒性**（去掉敏感阈值θ）；
- 开源模型：https://huggingface.co/niuzb/wav2vec2-ssd
- 方法论：验证了“短块处理+幂集分类”可解决传统幂集方法的“类数爆炸”问题，为后续研究提供范式。

### 什么是说话人语音分割（Speaker Diarization）？
将一段音频流按“说话人身份”分割为时间连续的同质片段（例如：“0-5s：说话人A”“5-12s：说话人B”“12-18s：A和B重叠”）。



#### 传统多阶段方法（经典框架）
流程：语音活动检测（VAD）->说话人嵌入（提取区分性特征）->无监督聚类（按嵌入分组）。  
核心缺陷：
- **误差传播**：前一步的错误会累积到后续步骤（例如VAD漏检会导致嵌入提取缺失）；
- **重叠语音处理差**：无法直接检测“多说话人同时说话”的片段，需额外后处理。

#### 端到端神经语音分割（EEND，2019年提出）
改进思路：用单个神经网络直接输入音频、输出分割结果，解决误差传播问题。  
实现方式：将任务建模为**帧级多标签分类**——每个说话人对应一个“是否活跃”的二分类标签，允许多个标签同时激活（例如“说话人A和B重叠”对应A、B标签均为1），并通过**置换不变训练（Permutation-Invariant Training, PIT）** 将无监督聚类转化为有监督任务（注：PIT的核心是“说话人标签无序”，需遍历所有标签置换，找最小损失的置换）。  
![permutaion free loss](../assets/images/permutation%20free%20loss.png)
EEND的局限：
- 难以预测“说话人数量”（测试集中数量超过训练时，性能骤降）；
- 数据饥渴：每个对话仅算1个训练样本，需依赖不真实的合成对话；
- 扩展性差：自注意力机制无法处理长对话。

#### 混合方法（Best of Both Worlds）
为解决EEND的局限，近年出现“多阶段+端到端”混合框架，流程为：
1. 长对话切分为短块（如5s）；
2. 用EEND处理每个短块（局部分割）；
3. 用说话人嵌入+无监督聚类拼接短块（全局整合）。
![hybrid method](../assets/images/hybrid%20method.png)

但**关键缺陷**：这类混合方法仍沿用EEND的“多标签分类”框架，未解决其核心问题——依赖“检测阈值（θ）”（需手动调优，对性能影响极大）、重叠语音预测精度低。


## 从多标签分类到幂集多分类
放弃“每个说话人独立二分类”的多标签思路，转而将“所有可能的说话人组合”作为互斥类别，即“幂集多分类”。


### 多标签分类回顾（基线框架）
以“局部最大说话人数量Kmax=3”为例：
- 输出维度：Kmax=3（每个维度对应一个说话人）；
- 标签编码：帧级二进制向量（如`[1,0,0]`表示“仅说话人S1活跃”，`[1,1,0]`表示“S1和S2重叠”）；
- 激活函数：Sigmoid（将每个维度输出压缩到[0,1]，表示“该说话人活跃的概率”）；
- 损失函数：置换不变二元交叉熵（PIT-BCE）——遍历3个说话人标签的所有置换，找最小BCE损失；
- 测试阶段：需设置**检测阈值θ**（如θ=0.5），输出概率超过θ则判定为“活跃”（θ需手动调优，敏感且依赖数据）。


### 幂集多分类设计
#### 核心思想
将“所有可能的说话人组合（包括无语音）”视为**互斥的单类别**，覆盖所有场景。仍以Kmax=3为例：
- 幂集类别总数：\(2^{Kmax} - 1 = 7\)（忽略3个及以上重叠，因数据中占比<2%，共7类）：
  1. 0：无语音；
  2. {S1}、{S2}、{S3}：单个说话人活跃；
  3. {S1,S2}、{S1,S3}、{S2,S3}：两个说话人重叠。

#### 模型适配修改
仅需调整神经网络的输出层和损失，无需改变主干结构：
| 模块                | 多标签分类（基线） | 幂集多分类 |
|---------------------|--------------------|--------------------|
| 输出层维度          | Kmax=3             | Kpowerset=7        |
| 激活函数            | Sigmoid            | Softmax（类间互斥）|
| 损失函数            | 置换不变BCE        | 置换不变交叉熵（CE）|
| 测试阶段决策        | 依赖检测阈值θ      | 直接取argmax（无参数）|




## 4. 实验结果与关键发现
### 4.1 核心性能对比（DER，越低越好）
| 方法                  | 领域内平均DER | DIHARD III（领域不匹配）DER |
|-----------------------|---------------|------------------------------|
| 预训练多标签基线      | 23.7%         | 25.9%                        |
| 多标签（复合训练）    | 25.6%         | 33.8%                        |
| 多标签（复合+领域适配）| 23.0%         | 22.4%                        |
| **幂集（复合训练）**  | **23.5%**     | **29.9%**                    |
| **幂集（复合+领域适配）** | **21.7%**  | **21.3%**                    |

关键结论：
1. 幂集方法**无需领域适配**已优于多标签（复合训练下，领域内DER从25.6%->23.5%，8%相对提升）；
2. 领域不匹配场景下，幂集提升更显著（DER从33.8%->29.9%，11%相对提升），因去掉了敏感阈值θ；
3. 幂集+领域适配后，在AISHELL-4、AliMeeting等6个数据集上达**SOTA**。


### 为什么幂集更好？
通过DIHARD III的误差拆解（图3），发现提升主要来自：
- **重叠语音漏检减少**：多标签的重叠漏检率13.1%->幂集9.9%（因幂集将“重叠”作为显式类别建模，而非依赖两个标签的阈值判断）；
- 假阳性（False Alarm）和说话人混淆（Speaker Confusion）无显著变化。


### 领域适配的额外价值
幂集模型在各DIHARD子领域（如餐厅、临床、广播）微调后，能更好平衡“假阳性”和“漏检”，且不影响说话人混淆率——相当于用领域适配替代了多标签中“手动调θ”的作用，进一步提升鲁棒性。





## 关键术语补充
- **DER（Diarization Error Rate）**：语音分割错误率，衡量“错分片段时长+漏检语音时长+假阳性语音时长”占总语音时长的比例，是该领域核心指标；
- **置换不变训练（PIT）**：因说话人标签无固定顺序（如S1和S2互换不改变语义），需通过遍历标签置换找最小损失，确保模型不依赖标签顺序；
- **幂集（Powerset）**：数学概念，指一个集合的所有子集构成的集合，此处用于“所有说话人组合”的类别定义。
## 为什么损失函数重要
从实验结果来看，新的损失函数在重叠语音上的性能提升了20%，在非重叠语音上的性能提升了10%。参考人脸识别领域的模型演变历史,新损失函数的发明都带来了模型性能的提升.
人脸识别模型的损失函数经历了多个发展阶段，以不断提升人脸识别的性能。下面按照时间顺序梳理其演进历史：
1. **Softmax loss**：经典的分类损失函数，将正确类别的预测概率最大化，广泛应用于图像分类领域。但它只考虑能否正确分类，未考虑类间距离，在人脸识别任务中，仅具有可分离性，缺乏判别性，不能实现类内聚合 ，难以满足人脸识别对特征的要求。
2. **Center loss**：2016年提出，在Softmax loss基础上增加了让同类样本向中心靠拢的项，为每个类设置一个中心，在保证分类的同时最小化类内距离。不过它本身没有分类功能，需配合Softmax loss使用。中心初始化是随机值，之后随学习到的特征实时更新。计算每一类的中心损失时要除以该类样本数计算均值，以防止样本失衡导致不同类别梯度更新不同步，人脸识别中参数经验值一般取0.003。其存在类内距优化效果不理想、类别多时对硬件要求高、L2范数的离群点难以优化以及只适用于同类样本间差异较小的数据等问题。
3. **Triplet loss**：三元组损失函数，由Anchor、Negative、Positive组成。通过使Anchor和Positive尽量靠近（减小同类距离），Anchor和Negative尽量远离（增大不同类间距离）来优化模型。在Center loss之前是人脸识别的常用损失函数，但样本数增多时，样本对的组合数量会指数级激增，训练耗时严重。
4. **L-softmax loss**：调整Softmax Loss，将卷积运算转化为向量积，把\(cosθ\)改成\(cos(mθ)\)（\(m>1\)），增加决策余量，加大学习难度，从而压缩类内距增大类间距。
5. **SphereFace（A - Softmax loss）**：2017年提出 ，在L-softmax loss基础上将权重归一化，使特征上的点映射到单位超球面上，模型的预测仅取决于\(W\)和\(X\)之间的角度。提出角度间隔惩罚，让训练更集中在优化深度特征映射和特征向量角度上，降低样本数量不均衡问题，但损失函数计算需一系列近似，导致网络训练不稳定。
6. **CosFace（AM - Softmax loss）**：2018年提出，直接将cosine间隔惩罚添加到目标逻辑回归中，采用加法余弦间隔（\(cos(θ)-m\)），归一化特征向量和权重。相比SphereFace，性能更好、实现更容易，减少了对softmax loss联合监督的需求。
7. **ArcFace**：2018年提出加性角度间隔损失（\(θ+m\)），同样归一化特征向量和权重，几何上有恒定的线性角度margin，直接优化弧度。为保证性能稳定，不需要与其他loss函数联合监督，在人脸识别任务中表现优异。 


