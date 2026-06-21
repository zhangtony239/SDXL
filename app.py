from dotenv import load_dotenv
load_dotenv()

import torch
import duckdb
import numpy as np
import transformers
from diffusers.utils import logging
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from torch.amp.autocast_mode import autocast
from compel import CompelForSDXL
from random import randint
from tags.TagCompleter import TagCompleter
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

# --- 配置区 ---
ckpt_path = 'miaomiaoRealskin_vPredV11.safetensors'
output_path = 'C:\\Users\\Tony\\Downloads\\'
negative_prompt = 'worst quality,bad quality,simple_background,low quality,jpeg artifacts,old,oldest,signature,shiny_skin,bad hands,bad feet,'
hotwords = {
    'airki': '1girl,white hair,blue eyes,cat ears',
    }
# --- 配置区 ---

# 简化diffusers日志
logging.disable_progress_bar()
transformers.utils.logging.set_verbosity_error()

# TF32计算加速
torch.set_float32_matmul_precision('high')

pipe = StableDiffusionXLPipeline.from_single_file(
    ckpt_path,
    use_safetensors=True,
    disable_mmap=True,
    torch_dtype=torch.float16
)
scheduler_args = {'prediction_type': 'v_prediction', 'rescale_betas_zero_snr': True}
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, **scheduler_args)
pipe.vae.enable_tiling() # 解锁更高分辨率
pipe.vae.to(torch.float32)
pipe = pipe.to('xpu', memory_format=torch.channels_last)

compel = CompelForSDXL(pipe=pipe)
history = InMemoryHistory()

gen = torch.Generator(device='cpu')
MAX_SEED = np.iinfo(np.int32).max

def draw(prompt,seed):
    conditioning = compel(prompt, negative_prompt=negative_prompt)

    print(f'Current seed: {seed}')

    with torch.inference_mode():
        # Ksample去噪阶段
        latent = pipe(
            prompt_embeds=conditioning.embeds,
            pooled_prompt_embeds=conditioning.pooled_embeds,
            negative_prompt_embeds=conditioning.negative_embeds,
            negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
            clip_skip=2,
            width=1024,
            height=1536,
            num_inference_steps=30,
            guidance_scale=5,
            generator=gen.manual_seed(seed),
            output_type='latent'
        ).images # type: ignore # 这里的images实则返回的是latent内容

        # VAE解码阶段
        latent = latent / pipe.vae.config.scaling_factor
        with autocast(device_type='xpu'):
            image_tensor = pipe.vae.decode(latent).sample
        image = pipe.image_processor.postprocess(image_tensor)[0] # type: ignore
        
    image.save(f'{output_path}{seed}.png') # type: ignore


if __name__ == '__main__':
    print('Loading tags database...')
    try:
        db = duckdb.connect()
        query = '''
            SELECT tag FROM (
                SELECT character AS tag, count FROM read_csv('tags/danbooru_character.csv', ignore_errors=true) WHERE count >= 100
                UNION ALL
                SELECT character AS tag, count FROM read_csv('tags/e621_character.csv', ignore_errors=true) WHERE count >= 100
                UNION ALL
                SELECT tags AS tag, count FROM read_csv('tags/danbooru_e621_merged.csv', ignore_errors=true) WHERE count >= 100
            )
            GROUP BY tag
            ORDER BY MAX(count) DESC
        '''
        results = db.execute(query).fetchall()
        tags = [row[0] for row in results]
        # 合并配置区 hotwords 的 key，并去重（保持 hotwords 优先级最高）
        hotword_keys = list(hotwords.keys())
        tags = hotword_keys + [tag for tag in tags if tag not in hotwords]
        print(f'Loaded {len(tags)} tags for completion (including {len(hotword_keys)} hotwords).')
    except Exception as e:
        print(f'Failed to load tags database: {e}')
        # 如果加载失败，至少保留 hotwords 的 key
        tags = list(hotwords.keys())
        
    completer = TagCompleter(tags)
    
    while True:
        try:
            prompts = prompt('Prompt: ', completer=completer, history=history).strip()
        except KeyboardInterrupt:
            continue
        if prompts in ['Q','q','exit']:
            break
        elif len(prompts.split('seed')) > 1:
            seed = int(prompts.split('seed')[1].split(',')[0][1:])
            prompts = prompts.split('seed')[0] + ','.join(prompts.split('seed')[1].split(',')[1:])
        else:
            seed = randint(0, MAX_SEED)
        prompt_tags = [t.strip() for t in prompts.split(',')]
        processed_tags = [hotwords[t.lower()] if t.lower() in hotwords else t for t in prompt_tags]
        prompts = ','.join(processed_tags)
        try:
            draw(prompts, seed)
        except KeyboardInterrupt:
            print('Drawing cancelled.')
            continue