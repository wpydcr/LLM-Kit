import gradio as gr

theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.sky,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.sky,
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        input_background_fill="#F6F6F6"
    )
