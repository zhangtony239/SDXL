import torch
import numpy as np
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from compel import CompelForSDXL
from random import randint

# --- 配置区 ---
#ollama_path = "C:\\Users\\Tony\\应用\\ollama-ipex-llm\\"
ckpt_path = "miaomiaoRealskin_vPredV11.safetensors"
output_path = "C:\\Users\\Tony\\Downloads\\"
negative_prompt = "worst quality,bad quality,simple_background,low quality,jpeg artifacts,old,oldest,signature,shiny_skin,bad hands,bad feet,"
hotwords = {
    'AIRKI': '1girl,white hair,blue eyes,cat ears',
    }
# --- 配置区 ---

'''Ollama 暂不常驻
ollama_info = subprocess.run(ollama_path+"ollama ps", text=True, capture_output=True).stdout
if len(ollama_info.splitlines()) > 1:
    print("Detected Ollama's model alive, stopping...")
    for info in ollama_info.splitlines()[1:]:
        name = info.split(' ')[0]
        subprocess.run(ollama_path+"ollama stop "+name)
        print('Stopped '+name)
'''
def vae_forward_wrapper(original_forward):
    def wrapper(sample, *args, **kwargs):
        # 强制将输入转为 float32
        return original_forward(sample.to(dtype=torch.float32), *args, **kwargs)
    return wrapper

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

    image = pipe(
        prompt_embeds=conditioning.embeds,
        pooled_prompt_embeds=conditioning.pooled_embeds,
        negative_prompt_embeds=conditioning.negative_embeds,
        negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
        width=1024,
        height=1536,
        num_inference_steps=30,
        guidance_scale=5,
        generator=torch.Generator().manual_seed(seed),
    ).images[0] # pyright: ignore[reportAttributeAccessIssue]
    
    image.save(f"{output_path}{seed}.png")

if __name__ == "__main__":
    while True:
        prompt = input("Prompt: ").strip()
        if prompt in ['Q','q','exit']:
            break
        if len(prompt.split('seed')) > 1:
            seed = int(prompt.split('seed')[1].split(',')[0][1:])
            prompt = prompt.split('seed')[0] + ','.join(prompt.split('seed')[1].split(',')[1:])
        else:
            seed = randint(0, MAX_SEED)
        prompt_tags = [t.strip() for t in prompt.split(',')]
        processed_tags = [hotwords[t.upper()] if t.upper() in hotwords else t for t in prompt_tags]
        prompt = ",".join(processed_tags)
        draw(prompt, seed)