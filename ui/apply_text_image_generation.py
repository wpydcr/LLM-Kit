import gradio as gr
from utils.text_image_utils import TextImage

text_image = TextImage()

def get_index(evt: gr.SelectData):
    text_image.cur_index = evt.index + 1


def apply_text_image_generation(localizer):
    with gr.Tab(label=localizer("Midjourney")):
        with gr.Row():
            with gr.Column(scale=3):
                images = gr.Gallery(label=localizer("图片"),columns=4)
                sub_image = gr.Image(label=localizer("子图"))
                prompt = gr.Textbox(label=localizer("提示词"))
                with gr.Accordion(label=''):
                    generate_btn = gr.Button(localizer("生成"))
                    upscale_btn = gr.Button(localizer("放大"),variant='primary')
            with gr.Column(scale=2):
                channel_id = gr.Textbox(label=localizer("Channel ID"))
                authorization = gr.Textbox(label=localizer("Authorization"))
                application_id = gr.Textbox(label=localizer("Application ID"))
                guild_id = gr.Textbox(label=localizer("Guild ID"))
                session_id = gr.Textbox(label=localizer("Session ID"))
                version = gr.Textbox(label=localizer("Version"))
                id = gr.Textbox(label=localizer("ID"))
                flags = gr.Textbox(value="--v 5.2", label=localizer("Flags"))
                with gr.Accordion(label=''):
                    set_config_btn = gr.Button(localizer("设置"))

    generate_btn.click(text_image.get_whole_image,inputs=[prompt],outputs=[images],show_progress=True)
    set_config_btn.click(text_image.setv,inputs=[channel_id,authorization,application_id,guild_id,session_id,version,id,flags],show_progress=True)
    images.select(get_index,show_progress=True)
    upscale_btn.click(text_image.upscale,outputs=[sub_image],show_progress=True)
