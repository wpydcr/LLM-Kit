import gradio as gr
from ui.apply_video import Avtar
from ui.apply_knowledge import apply_knowledge

def apply_page(localizer):
    with gr.Tab(localizer("角色扮演")) as avatar_tab:
        avatar = Avtar()
        avatar_tab.select(avatar.apply_video(localizer), outputs=[])
    with gr.Tab(localizer('知识库')):
        apply_knowledge(localizer)