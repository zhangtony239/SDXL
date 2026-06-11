import torch
import numpy as np
from diffusers.utils import logging
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from compel import CompelForSDXL
from random import randint
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import Completer, Completion
import duckdb

# --- 配置区 ---
ckpt_path = "miaomiaoRealskin_vPredV11.safetensors"
output_path = "C:\\Users\\Tony\\Downloads\\"
negative_prompt = "worst quality,bad quality,simple_background,low quality,jpeg artifacts,old,oldest,signature,shiny_skin,bad hands,bad feet,"
hotwords = {
    'airki': '1girl,white hair,blue eyes,cat ears',
    }
# --- 配置区 ---

def vae_forward_wrapper(original_forward):
    def wrapper(sample, *args, **kwargs):
        # 强制将输入转为 float32
        return original_forward(sample.to(dtype=torch.float32), *args, **kwargs)
    return wrapper

# 简化diffusers日志
logging.disable_progress_bar()

# TF32计算加速
torch.set_float32_matmul_precision("high")

pipe = StableDiffusionXLPipeline.from_single_file(
    ckpt_path,
    use_safetensors=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)
scheduler_args = {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True}
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, **scheduler_args)
pipe.text_encoder.config.num_hidden_layers -= 2 # CLIP skip: 2
pipe.vae.decode = vae_forward_wrapper(pipe.vae.decode)
pipe.vae.enable_tiling() # 解锁更高分辨率
pipe.vae.to(torch.float32)
pipe = pipe.to("xpu")

# 设置内存格式为 channels_last
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.decoder.to(memory_format=torch.channels_last)

compel = CompelForSDXL(pipe=pipe)

MAX_SEED = np.iinfo(np.int32).max

def draw(prompt,seed):
    conditioning = compel(prompt, negative_prompt=negative_prompt)

    print(f"Current seed: {seed}")

    with torch.inference_mode():
        image = pipe(
            prompt_embeds=conditioning.embeds,
            pooled_prompt_embeds=conditioning.pooled_embeds,
            negative_prompt_embeds=conditioning.negative_embeds,
            negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
            width=1024,
            height=1536,
            num_inference_steps=30,
            guidance_scale=5,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0] # type: ignore
        
    image.save(f"{output_path}{seed}.png")

class CommaSeparatedCompleter(Completer):
    def __init__(self, words):
        self.words = words

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        parts = text.split(',')
        if not parts:
            return
        
        current_part = parts[-1]
        current_word = current_part.lstrip()
        start_position = -len(current_word)
        
        if not current_word:
            return
            
        count = 0
        prefix_matches = []
        
        # 1. 前缀匹配
        for word in self.words:
            if word.lower().startswith(current_word.lower()):
                prefix_matches.append(word)
                yield Completion(word, start_position=start_position)
                count += 1
                if count >= 20:
                    break
                    
        # 2. 包含匹配
        if count < 20:
            for word in self.words:
                if word not in prefix_matches and current_word.lower() in word.lower():
                    yield Completion(word, start_position=start_position)
                    count += 1
                    if count >= 20:
                        break

if __name__ == "__main__":
    print("Loading tags database...")
    try:
        db = duckdb.connect()
        query = """
            SELECT character FROM (
                SELECT character, count FROM 'tags/danbooru_character.csv' WHERE count >= 100
                UNION ALL
                SELECT character, count FROM 'tags/e621_character.csv' WHERE count >= 100
            )
            GROUP BY character
            ORDER BY MAX(count) DESC
        """
        results = db.execute(query).fetchall()
        tags = [row[0] for row in results]
        # 合并配置区 hotwords 的 key，并去重（保持 hotwords 优先级最高）
        hotword_keys = list(hotwords.keys())
        tags = hotword_keys + [tag for tag in tags if tag not in hotwords]
        print(f"Loaded {len(tags)} tags for completion (including {len(hotword_keys)} hotwords).")
    except Exception as e:
        print(f"Failed to load tags database: {e}")
        # 如果加载失败，至少保留 hotwords 的 key
        tags = list(hotwords.keys())
        
    completer = CommaSeparatedCompleter(tags)
    
    while True:
        try:
            prompt = pt_prompt("Prompt: ", completer=completer).strip()
        except KeyboardInterrupt:
            print()
            continue
        if prompt in ['Q','q','exit']:
            break
        elif len(prompt.split('seed')) > 1:
            seed = int(prompt.split('seed')[1].split(',')[0][1:])
            prompt = prompt.split('seed')[0] + ','.join(prompt.split('seed')[1].split(',')[1:])
        else:
            seed = randint(0, MAX_SEED)
        prompt_tags = [t.strip() for t in prompt.split(',')]
        processed_tags = [hotwords[t.lower()] if t.lower() in hotwords else t for t in prompt_tags]
        prompt = ",".join(processed_tags)
        draw(prompt, seed)