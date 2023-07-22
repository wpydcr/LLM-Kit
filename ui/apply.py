import gradio as gr
from ui.apply_video import Avtar
from ui.apply_knowledge import apply_knowledge

def apply_page():
    with gr.Tab("角色扮演") as avatar_tab:
        avatar = Avtar()
        avatar_tab.select(avatar.apply_video(), outputs=[])
    with gr.Tab('知识库'):
        apply_knowledge()