<p align="center"><img src="https://raw.githubusercontent.com/ikun-llm/.github/main/profile/logo.png" width="120" /></p>
<h2 align="center">ikun-DPO</h2>
<p align="center"><b>让模型学会什么回答更美</b><br/><sub>Level 2 | 对齐篇</sub></p>

---

> chosen = 鸡你太美的回答，rejected = 小黑子的回答。

## 你将学到

- 什么是偏好对齐？为什么 SFT 后还需要对齐？
- RLHF 太复杂 → DPO 的简化思路
- DPO 损失函数原理（不需要奖励模型！）
- 如何构造 chosen/rejected 偏好数据
- 训练前后效果对比

## 核心代码

基于 [MiniMind](https://github.com/jingyaogong/minimind) 的 `trainer/train_dpo.py`

```bash
cd trainer && python train_dpo.py \
    --from_weight full_sft --beta 0.1 \
    --hidden_size 512 --epochs 2
```

## DPO 直觉理解

```
传统 RLHF: 训练奖励模型 → PPO 强化学习 → 复杂！
DPO:      直接从偏好数据学 → 一步到位 → 简单！

核心思想: 好的回答概率 ↑，差的回答概率 ↓
Loss = -log(σ(β * (好的log概率 - 差的log概率)))
```

## 数据格式

```json
{
  "chosen": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！我是ikun，练习时长两年半！鸡你太美~"}
  ],
  "rejected": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好。"}
  ]
}
```

## 系列导航

| Level | Repo | 学什么 |
|-------|------|--------|
| 1 | [ikun-tokenizer](https://github.com/ikun-llm/ikun-tokenizer) | 分词器原理 |
| 1 | [ikun-pretrain](https://github.com/ikun-llm/ikun-pretrain) | 从零预训练 |
| 1 | [ikun-2.5B](https://github.com/ikun-llm/ikun-2.5B) | SFT + LoRA 微调 |
| **2** | **ikun-DPO** <-- 你在这里 | 偏好对齐 |
| 2 | [ikun-GRPO](https://github.com/ikun-llm/ikun-GRPO) | 强化学习 |
| 2 | [ikun-Reason](https://github.com/ikun-llm/ikun-Reason) | 推理模型 |
| 3 | [ikun-MoE](https://github.com/ikun-llm/ikun-MoE) | 混合专家 |
| 3 | [ikun-Distill](https://github.com/ikun-llm/ikun-Distill) | 知识蒸馏 |
| 3 | [ikun-V](https://github.com/ikun-llm/ikun-V) | 多模态 |
| 4 | [ikun-deploy](https://github.com/ikun-llm/ikun-deploy) | 部署 |
