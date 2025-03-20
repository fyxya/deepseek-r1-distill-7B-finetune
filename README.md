# DeepSeek-R1-Distill-7B-SFT-Project

本项目基于DeepSeek-R1-Distill-Qwen-7B模型进行角色扮演微调，实现与《Fate/stay night》中Saber（阿尔托莉雅）的角色对话功能。

## ✨ 功能特性
- **高效微调**：使用LoRA+4位量化技术，消费级显卡可训练
- **角色还原**：定制化对话数据集，高度还原Saber语言风格
- **生产部署**：支持全量模型导出与4位量化推理
- **对话模板**：Prompt: {用户输入} Completion: {角色回答}<|endoftext|>

- ## 📦 环境依赖
- pip install -r requirements.txt

## 🚀 快速开始
- 训练
- python main.py
- 推理
- python inference.py

## 📂 数据集结构
json {"prompt": "用户输入", "completion": "角色回答"}

## ⚙️ 训练配置
LoRA配置
LoraConfig( r=4, lora_alpha=8, lora_dropout=0.1, task_type=TaskType.CAUSAL_LM )
训练参数
TrainingArguments( num_train_epochs=10, per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=5e-5, fp16=True, optim="adamw_bnb_8bit" )

## 💾 模型保存
保存LoRA适配器
model.save_pretrained("./saved_lora_models")
合并并保存全量模型
merged_model = model.merge_and_unload() merged_model.save_pretrained("./final_saved_models")


