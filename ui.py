import gradio as gr
import numpy as np

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

css = '''
.tagpage{
    width: 100%;
    height: 90vh;
}
'''

tagpage = '''
<iframe class='tagpage' src='https://magic-tag.netlify.app/#'></iframe>
'''

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column(scale=7):
            with gr.Group():
                gr.HTML(value=tagpage)
        with gr.Column(scale=3):
            with gr.Group():
                with gr.Row():
                    prompt = gr.Text(
                        label="关键词",
                        show_label=True,
                        max_lines=5,
                        placeholder="输入你要的图片关键词",
                        container=False,
                    )
                    run_button = gr.Button("生成", scale=0, variant="primary")
                result = gr.Image(label="Result", show_label=False, format="png")
            with gr.Accordion("高级选项", open=False):
                with gr.Row():
                    use_negative_prompt = gr.Checkbox(label="使用反向词条", value=True)
                    negative_prompt = gr.Text(
                        label="反向词条",
                        max_lines=5,
                        lines=4,
                        placeholder="输入你要排除的图片关键词",
                        value="worst quality,bad quality,simple_background,low quality,jpeg artifacts,old,oldest,signature,shiny_skin,bad hands,bad feet,",
                        visible=True,
                    )
                seed = gr.Slider(
                    label="种子",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="随机种子", value=True)
                with gr.Row(visible=True):
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
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="提示词引导程度",
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

'''
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
'''

if __name__ == "__main__":
    demo.launch()