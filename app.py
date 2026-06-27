from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# --- 配置区 ---
ckpt_path = 'miaomiaoRealskin_vPredV11.safetensors'
output_path = 'C:\\Users\\Tony\\Downloads\\'
negative_prompt = 'worst quality,bad quality,simple_background,low quality,jpeg artifacts,old,oldest,signature,shiny_skin,bad hands,bad feet,'
hotwords = {
    'airki': '1girl,white hair,blue eyes,cat ears',
    }
# --- 配置区 ---

with tqdm(total=12, desc='Importing dependencies') as pbar:
    import torch
    pbar.update()
    import numpy as np
    pbar.update()
    import transformers
    pbar.update()
    from diffusers.utils import logging
    pbar.update()
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
    pbar.update()
    from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
    pbar.update()
    from torch.amp.autocast_mode import autocast
    pbar.update()
    from compel import CompelForSDXL
    pbar.update()
    from random import randint
    pbar.update()
    from tags.TagCompleter import init_tags
    pbar.update()
    from prompt_toolkit import prompt
    pbar.update()
    from prompt_toolkit.history import InMemoryHistory
    pbar.update()


# 简化diffusers日志
logging.disable_progress_bar()
transformers.utils.logging.set_verbosity_error()

# TF32计算加速
torch.set_float32_matmul_precision('high')

# 解决xpu异步太激进导致的tqdm脱钩问题
def xpu_sync_callback(*args, **kwargs):
    torch.xpu.synchronize()
    return args[-1]

print('Initializing pipeline...', end='')
pipe = StableDiffusionXLPipeline.from_single_file(
    ckpt_path,
    use_safetensors=True,
    disable_mmap=True,
    torch_dtype=torch.float16,
)
scheduler_args = {
    'prediction_type': 'v_prediction',
    'rescale_betas_zero_snr': True,
    'use_exponential_sigmas': True
}
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, **scheduler_args)
pipe = pipe.to('xpu', memory_format=torch.channels_last)

compel = CompelForSDXL(pipe=pipe)
history = InMemoryHistory()
completer = init_tags(hotwords=hotwords)

gen = torch.Generator(device='xpu')
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
            callback_on_step_end=xpu_sync_callback,
            clip_skip=2,
            width=1024,
            height=1536,
            num_inference_steps=30,
            guidance_scale=3.8,
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