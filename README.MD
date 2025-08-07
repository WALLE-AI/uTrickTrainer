# 各种大模型技术训练trick和样例训练（站在巨人肩膀打造领域多模态模型）
* 提供各种开源数据集清洗过滤和合成策略
* 提供全栈训练策略，包括语言大模型、多模态大模型等
## vision encoder
* 支持主流的vision encoder，其中包括clip-vit-base-patch16、Qwen-VIT、Intern-VIT、MetaClip Sgclip等等(一种直接从现有视觉大模型中直接抽取VIT部分作为视觉部分)
## 连接层设计
* cross attention
* MLP
* Q-former
* ....
## LLM部分
* 一键式适配不同主流的模型，比如Qwen系列、Intern系列
* 也通过LLM模型Head层的改造