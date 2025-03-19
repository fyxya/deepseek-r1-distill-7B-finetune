import gc
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
from transformers import TrainingArguments, Trainer

# 第一步 载入预训练模型
model_name = "DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("模型加载完毕")


# 第三步 生成训练集和测试集

raw_dataset = load_dataset("json", name="train", data_files="datasets.jsonl")
dataset = raw_dataset["train"].train_test_split(
    test_size=0.1,
    shuffle=True,
    seed=42
)

train_dataset = dataset["train"]
test_dataset = dataset["test"]
print("训练数据准备完毕")


# 第四步 准备tokenizer
def tokenizer_function(sam):
    texts = [f"Prompt: {p}\nCompletion: {c}<|endoftext|>"
             for p, c in zip(sam["prompt"], sam["completion"])]
    tokens = tokenizer(
        texts,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    # 创建标签并mask填充部分
    labels = tokens.input_ids.clone()
    labels[tokens.attention_mask == 0] = -100  # 忽略填充部分的loss计算
    tokens["labels"] = labels

    return tokens


tokenized_train_dataset = train_dataset.map(tokenizer_function, batched=True)
tokenized_eval_dataset = test_dataset.map(tokenizer_function, batched=True)
print("完成数据词元化")

# 第五步 量化设置

quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16,  # 计算时使用 float16 加速
                                        bnb_4bit_use_double_quant=True,       # 启用双量化进一步压缩模型
                                        bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
print("完成量化模型加载")

# 第六步 lora设置
lora_config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.1, task_type=TaskType.CAUSAL_LM)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.config.use_cache = False  # 显式禁用缓存
model.enable_input_require_grads()  # 启用梯度追踪
model.gradient_checkpointing_enable()  # 显式启用梯度检查点

print("lora设置完成")

# 第七步 设置训练参数

training_args = TrainingArguments(
    output_dir="./finrtuned_model",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=10,

    learning_rate=5e-5,
    logging_dir="./logs",
    run_name="deepseek-r1-distill-finetune",
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    optim="adamw_bnb_8bit",
    remove_unused_columns=False,
    label_names=["labels"]
)
print("训练参数设置完成")

# 第八步 定义训练器 并 训练

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

torch.cuda.empty_cache()
print("------开始训练------")
trainer.train()
print("------训练完成------")

# 第九步 保存lora模型

save_path="./saved_lora_models"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("lora模型保存完成")


# 保存全量模型

final_save_path = "./final_saved_models"
save_path = "./saved_lora_models"

# 分阶段加载与合并
with torch.device("cpu"):  # 强制在CPU执行
    # 阶段1：仅加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},  # 强制使用CPU
        offload_folder="./offload",
        trust_remote_code=True
    )

    # 阶段2：加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        save_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16
    )


    # 阶段3：合并模型（内存优化）
    merged_model = model.merge_and_unload()
    del base_model, model  # 立即释放内存
    gc.collect()
    torch.cuda.empty_cache()

    # 阶段4：分片保存
    merged_model.save_pretrained(
        final_save_path,
        safe_serialization=True,
        max_shard_size="2GB",  # 适应小内存环境
        state_dict_split=True
    )

# 单独保存tokenizer
tokenizer.save_pretrained(final_save_path)
print("全量模型保存完成")