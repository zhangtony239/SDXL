# pyright: reportMissingImports=false
import random
import gradio as gr
import numpy as np
import spaces
import torch
import os
from huggingface_hub import hf_hub_download
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from compel import CompelForSDXL
from PIL import Image, PngImagePlugin
import io

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

hotwords = {
    'AIRKI':'1girl,white hair,blue eyes,cat ears',
    }

# Add metadata to the image
def add_metadata_to_image(image, metadata):
    # 构建符合 A1111 规范的字符串，这样别人下载你的图能直接“一键导入”
    meta_text = f"{metadata['prompt']}\nNegative prompt: {metadata['negative_prompt']}\n"
    meta_text += f"Steps: {metadata['num_inference_steps']}, Sampler: Eular a, "
    meta_text += f"CFG scale: {metadata['guidance_scale']}, Seed: {metadata['seed']}, "
    meta_text += f"Size: {metadata['width']}x{metadata['height']}, Model: {metadata['model']}"
    
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("parameters", meta_text)
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=png_info)
    buffer.seek(0)
    return Image.open(buffer)

def vae_forward_wrapper(original_forward):
    def wrapper(sample, *args, **kwargs):
        # 强制将输入转为 float32
        return original_forward(sample.to(dtype=torch.float32), *args, **kwargs)
    return wrapper

if not torch.cuda.is_available():
    DESCRIPTION = "\n<p>你现在运行在CPU上 但是此项目只支持GPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

if torch.cuda.is_available():
    model_path = hf_hub_download(
        repo_id="dsfkjlweuyr/miaomiaoRealskin",  # 模型仓库名称（非完整URL）
        filename="miaomiaoRealskin_vPredV11.safetensors",
        token=os.environ.get("HF_TOKEN")
    )
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    scheduler_args = {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True}
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe.text_encoder.config.num_hidden_layers -= 2 # CLIP skip: 2
    pipe.vae.decode = vae_forward_wrapper(pipe.vae.decode)
    pipe.vae.to(torch.float32)
    pipe.to("cuda")

    # 设置内存格式为 channels_last
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.decoder.to(memory_format=torch.channels_last)

    # 初始化 Compel
    compel = CompelForSDXL(pipe=pipe)

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

@spaces.GPU(duration=10)
def infer(
    prompt: str,
    negative_prompt: str = "worst quality,bad quality,simple_background,low quality,jpeg artifacts,old,oldest,signature,shiny_skin,bad hands,bad feet,",
    use_negative_prompt: bool = True,
    seed: int = 0,
    width: int = 1024,
    height: int = 1536,
    guidance_scale: float = 5,
    num_inference_steps: int = 30,
    randomize_seed: bool = True,
    use_resolution_binning: bool = True,
    _=gr.Progress(track_tqdm=True),
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)
    
    if not use_negative_prompt:
        negative_prompt = ""
    
    original_prompt = prompt

    # 改进后的逻辑：只针对独立的 Tag 进行精确替换
    prompt_tags = [t.strip() for t in prompt.split(',')]
    processed_tags = [hotwords[t.upper()] if t.upper() in hotwords else t for t in prompt_tags]
    prompt = ",".join(processed_tags)
    
    with torch.inference_mode():
        conditioning = compel(prompt, negative_prompt=negative_prompt)
        
        # 在调用 pipe 时，使用新的参数名称（确保参数名称正确）
        image = pipe(
            prompt_embeds=conditioning.embeds,
            pooled_prompt_embeds=conditioning.pooled_embeds,
            negative_prompt_embeds=conditioning.negative_embeds,
            negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            use_resolution_binning=use_resolution_binning,
        ).images[0] # pyright: ignore[reportAttributeAccessIssue]

    # Create metadata dictionary
    metadata = {
        "prompt": original_prompt,
        "processed_prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "model": "miaomiaoRealSkin11",
        "use_resolution_binning": use_resolution_binning,
    }
    # Add metadata to the image
    image_with_metadata = add_metadata_to_image(image, metadata)
    
    return image_with_metadata, seed

css = '''
::-webkit-scrollbar {
    display: none !important;
}
* {
    -ms-overflow-style: none !important;  /* IE and Edge */
    scrollbar-width: none !important;  /* Firefox */
}
footer{
    display: none !important;
}
.tagpage{
    width: 100%;
    height: 95vh;
}
.submit{
    min-width: 6em;
}
'''

tagpage = '''
<iframe class='tagpage' src='https://magic-tag.netlify.app/#'></iframe>
'''

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=7):
            with gr.Group():
                gr.HTML(value=tagpage)
        with gr.Column(scale=3):
            with gr.Group():
                with gr.Row(equal_height=True):
                    prompt = gr.Text(
                        max_lines=5,
                        placeholder="输入你要的图片关键词",
                        container=False,
                        scale=9,
                    )
                    run_button = gr.Button("生成", scale=1, elem_classes='submit', variant="primary")
                result = gr.Image(label="Result", show_label=False, format="png")
            with gr.Accordion("高级选项", open=False):
                with gr.Group():
                    use_negative_prompt = gr.Checkbox(label="使用反向词条", value=True)
                    negative_prompt = gr.Text(
                        container=False,
                        max_lines=5,
                        placeholder="输入你要排除的图片关键词",
                        value="worst quality,bad quality,simple_background,low quality,jpeg artifacts,old,oldest,signature,shiny_skin,bad hands,bad feet,",
                    )
                with gr.Column():
                    width = gr.Slider(
                        label="宽度",
                        minimum=512,
                        maximum=MAX_IMAGE_SIZE,
                        step=64,
                        value=1024,
                    )
                    height = gr.Slider(
                        label="高度",
                        minimum=512,
                        maximum=MAX_IMAGE_SIZE,
                        step=64,
                        value=1536,
                    )
                seed = gr.Slider(
                    label="种子",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="随机种子", value=True)
                with gr.Column():
                    guidance_scale = gr.Slider(
                        label="引导强度",
                        minimum=0.1,
                        maximum=10,
                        step=0.1,
                        value=5.0,
                    )
                    num_inference_steps = gr.Slider(
                        label="生成步数",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=30,
                    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
    )

    gr.on(
        triggers=[prompt.submit, run_button.click],
        fn=infer,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            randomize_seed,
        ],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch(
        css=css,
        theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Google Sans Flex"), "system-ui"]),
        ssr_mode=True,
        share=False,
    )