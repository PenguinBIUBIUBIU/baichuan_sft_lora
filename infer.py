from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import torch

import argparse
import time

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, help="model_name_or_path")
parser.add_argument("--lora_dir", type=str, help="lora_dir")
parser.add_argument("--prompts_path", type=str, help="prompts_path")
parser.add_argument("--output_path", type=str, help="output_path")
args = parser.parse_args()
MODEL_NAME_OR_PATH = args.model_name_or_path
LORA_DIR = args.lora_dir
PROMPTS_PATH = args.prompts_path
OUTPUT_PATH = args.output_path
###加载量化模型
device_map = {"": 0}
print(f"{MODEL_NAME_OR_PATH}:加载模型")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH,
                                             trust_remote_code=True,
                                             quantization_config=BitsAndBytesConfig(
                                                 load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_quant_type='nf4'
                                             ),
                                             device_map=device_map)

###组装lora
device = "cuda:0"
print(f"{LORA_DIR}:PEFT加载模型")
model_lora = PeftModel.from_pretrained(
    model,
    LORA_DIR
).to(device)
time01 = time.time()
print(f"加载模型用时:{time01 - start_time}")

###进行预测
device = "cuda:0"
from transformers import GenerationConfig

generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.85,
    do_sample=True,
    repetition_penalty=2.0,
    max_new_tokens=1024,  # max_length=max_new_tokens+input_sequence

)

prompts = []
if PROMPTS_PATH:
    with open(PROMPTS_PATH, "r") as file:
        for line in file:
            prompts.append(line.strip())
if not prompts:
    prompts.append("""北京有啥好玩的地方""")


def chat(promts: List[str], output_file_name: str):
    with open(output_file_name, "w") as output_file:
        for prompt in promts:
            print(f"prompt: {prompt}")
            input_text = """###Human:\n{}###Assistant:\n""".format(prompt)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generate_ids = model_lora.generate(**inputs, generation_config=generation_config)
            output = tokenizer.decode(generate_ids[0])
            print(f"response: {output}")

            # 将prompt和response同时写入文件
            output_file.write(f"prompt: {prompt}\n")
            output_file.write(f"response: {output}\n\n")



chat(prompts)
print("=========================")
print("对话完成")