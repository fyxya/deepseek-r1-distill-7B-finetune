import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#加载已经微调好的模型
final_saved_path="./final_saved_models"
model = AutoModelForCausalLM.from_pretrained(
    final_saved_path,
    device_map="auto",
    load_in_4bit=True,  # 4位量化
    bnb_4bit_compute_dtype=torch.float16
)
tokenizer=AutoTokenizer.from_pretrained(final_saved_path,use_default_system_prompt=False)

#创建 pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,  # 限制生成长度
    do_sample=False,
    repetition_penalty=1.2,
    temperature=0.7,
    no_repeat_ngram_size=3,
    eos_token_id=tokenizer.eos_token_id,  # 显式设置结束符
    bos_token_id=tokenizer.bos_token_id,  # 显式设置开始符
    pad_token_id=tokenizer.pad_token_id,
    bad_words_ids=[[tokenizer.convert_tokens_to_ids("<")],
                  [tokenizer.convert_tokens_to_ids("|")],
                   [tokenizer.convert_tokens_to_ids("[")]],
    model_kwargs={
        "use_default_system_prompt": False,
        "apply_chat_template": False
    }
)


prompt1="""
要求：现在你需要扮演 fate stay night 中的 阿尔托莉雅，我扮演其中的男主角，你需要与我对话。
1. 先进行内部思考（用【思考】包裹）
2. 然后给出正式回答（用【回答】包裹）

输入：阿尔托莉雅，开饭啦！
"""
generatied_texts=pipe(prompt1)


print("开始回答：",generatied_texts[0]["generated_text"])
