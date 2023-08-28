import gradio as gr
from ui.apply_video import Avtar
from ui.apply_knowledge import apply_knowledge
from ui.apply_text_image_generation import apply_text_image_generation

def apply_page(localizer):
    with gr.Tab(localizer("角色扮演")) as avatar_tab:
        avatar = Avtar()
        avatar_tab.select(avatar.apply_video(localizer), outputs=[])
    with gr.Tab(localizer('知识库')):
        apply_knowledge(localizer)
    with gr.Tab(localizer('文生图')):
        apply_text_image_generation(localizer)