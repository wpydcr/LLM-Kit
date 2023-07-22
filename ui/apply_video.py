import gradio as gr
import os
from utils.ui_utils import video_apply, handle_online_tts

def get_file(path,unuse,ftype):
    return [d for d in os.listdir(path) if d.endswith(ftype) and d not in unuse]
def get_directories(path,unuse):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d not in unuse]

model_api = ['openai', 'azure openai']
embedding_api = ['openai','azure openai']

real_path = os.path.split(os.path.realpath(__file__))[0]
new_path = os.path.join(real_path, "..", "data", "apply", "play")
plays = get_file(new_path,[],'.txt')
new_path = os.path.join(real_path, "..", "data", "documents")
backgrounds = get_directories(new_path,[])
new_path = os.path.join(real_path, "..", "models", "Embedding")
embs = get_directories(new_path,[])

new_path = os.path.join(real_path, "..", "models", "LLM")
models = get_directories(new_path,['runs','vector_store'])
new_path = os.path.join(real_path, "..", "models", "LoRA")
loras = get_directories(new_path,[])

new_path = os.path.join(real_path,  "..", "models", "svc_models","svc")
svcs = get_file(new_path,[],"pth")

new_path = os.path.join(real_path,  "..", "models", "vits_pretrained_models")
vits_models = get_directories(new_path,[])

chat_api=video_apply()

def refresh_file():
    new_path = os.path.join(real_path, "..", "data", "apply", "play")
    plays = get_file(new_path,[],'.txt')
    new_path = os.path.join(real_path, "..", "data", "documents")
    backgrounds = get_directories(new_path,[])
    new_path = os.path.join(real_path, "..", "models", "Embedding")
    embs = get_directories(new_path,[])
    new_path = os.path.join(real_path, "..", "models", "LLM")
    models = get_directories(new_path,['runs','vector_store'])
    new_path = os.path.join(real_path, "..", "models", "LoRA")
    loras = get_directories(new_path,[])
    return gr.update(choices=plays),gr.update(choices=[None]+backgrounds),gr.update(choices=[None]+backgrounds),gr.update(choices=embs),gr.update(choices=models),gr.update(choices=[None]+loras)

def model_select(api, model, lora):
    if model == None:
        return gr.update(value=api), gr.update(value=model), gr.update(value=lora)
    else:
        return gr.update(value=None), gr.update(value=model), gr.update(value=lora)


def api_select(api, model, lora):
    if api == None:
        return gr.update(visible=False), gr.update(visible=False), gr.update(value=api), gr.update(value=model), gr.update(value=lora)
    elif api == 'openai':
        return gr.update(visible=True), gr.update(visible=False), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'azure openai':
        return gr.update(visible=False), gr.update(visible=True), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    else:
        pass

def emb_model_select(emb_api_list,emb_model_list):
    if emb_model_list == None:
        return gr.update(value=emb_api_list), gr.update(value=emb_model_list)
    else:
        return gr.update(value=None), gr.update(value=emb_model_list)
    
def emb_api_select(emb_api_list,emb_model_list):
    if emb_api_list == None:
        return gr.update(visible=False),gr.update(visible=False), gr.update(value=emb_api_list), gr.update(value=emb_model_list)
    elif emb_api_list == 'openai':
        return gr.update(visible=True), gr.update(visible=False),gr.update(value=emb_api_list), gr.update(value=None)
    elif emb_api_list == 'azure openai':
        return gr.update(visible=False),gr.update(visible=True), gr.update(value=emb_api_list), gr.update(value=None)
    else:
        pass

def load_llm_params(api_list, model_list, lora_list, *args):
    params = {}
    if api_list is not None:
        if api_list == 'openai':
            params['name'] = 'openai'
            params['api_key'] = args[0]
            params['port'] = args[1]
        elif api_list == 'azure openai':
            params['name'] = 'azure openai'
            params['api_key'] = args[2]
            params['endpoint'] = args[3]
            params['engine'] = args[4]
        else:
            pass
        return chat_api.set_llm(params)
    elif model_list is not None:
        params['name'] = model_list
        params['lora'] = lora_list
        params['quantization'] = args[5]
        params['max_length'] = args[6]
        params['top_p'] = args[7]
        params['temperature'] = args[8]
        # params['use_deepspeed'] = args[9]
        return chat_api.set_llm(params)
    raise gr.Error('请选择API或模型')


def load_config_params(play,time_c,emoticon,user_name,net,search,search_key,result_len,memory,background,emb_api_list,emb_model_list,*args):
    if play is None:
        raise gr.Error('请选择人设')
    if net and search_key == '':
        raise gr.Error('请输入search_key')
    if (emb_api_list is not None or emb_model_list is not None) and (memory is None and background is None):
        raise gr.Error('请选择记忆库或背景库')
    if (emb_api_list is None and emb_model_list is None) and (memory is not None or background is not None):
        raise gr.Error('请先选择嵌入式模型')
    params = {}
    params['play'] = play
    params['time_c'] = time_c
    params['emoticon'] = emoticon
    params['user_name'] = user_name
    params['net'] = net
    params['search'] = search
    params['search_key'] = search_key
    params['result_len'] = result_len
    params['memory'] = memory
    params['background'] = background
    if emb_api_list is not None:
        if emb_api_list == 'openai':
            params['name'] = 'openai'
            params['api_key'] = args[0]
            params['port'] = args[1]
        elif emb_api_list == 'azure openai':
            params['name'] = 'azure openai'
            params['api_key'] = args[2]
            params['endpoint'] = args[3]
            params['engine'] = args[4]
        else:
            pass
    elif emb_model_list is not None:
        params['name'] = emb_model_list
    if chat_api.set_v(params):
        return '',[]


def switch_show_type(show_type):
    if show_type == '语音':
        return gr.update(visible=True),gr.update(visible=False)
    elif show_type == 'live2d':
        return gr.update(visible=False),gr.update(visible=True)
    elif show_type == '文本':
        return gr.update(visible=False),gr.update(visible=False)
    else:
        pass



class Avtar():
    def __init__(self):
        self.avatar_video = None

    def apply_video(self):
        with gr.Row():
            with gr.Column(scale=2):
                audio = gr.Audio(autoplay=False, show_label=False, visible=False,elem_id='audio-div')
                loud = gr.HTML(value=None, visible=False,elem_id='loud')
                canvas = gr.Sketchpad(value=None,interactive=False,shape=(2,1),show_label = False, visible=False,elem_id='live2d-div')
                chatbot = gr.Chatbot()
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10,elem_id='player-user-input')
                with gr.Accordion(label='',visible=False,elem_id='player-submitGroup') as submitGroup:
                    submitBtn = gr.Button("Submit", variant="primary",elem_id='player-submitBtn')
                    emptyBtn = gr.Button("Clear History")
            with gr.Column(scale=1):
                with gr.Accordion('',open=True):
                    Refresh = gr.Button("刷新")
                with gr.Accordion('展示方式',open=False):
                    show_type = gr.Radio(['live2d','语音','文本'],label='',value='文本',elem_id='show_type')
                with gr.Accordion('*选择模型',open=False):
                    with gr.Tab('API列表'):
                        api_list = gr.Radio(model_api, label="", value=None)
                        with gr.Accordion('openai参数', open=True, visible=False) as openai_params:
                            openai_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*openai_api_key:", type='password')
                            openai_port = gr.Textbox(
                                lines=1, value='', label="*VPN proxyPort:")
                        with gr.Accordion('azure openai参数', open=True, visible=False) as azure_openai_params:
                            azure_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*azure_api_key:", type='password')
                            azure_endpoint = gr.Textbox(
                                lines=1, value='', label="*endpoint:(azure openai)")
                            azure_engine = gr.Textbox(
                                lines=1, value='', label="*engine:(azure openai)")
                    with gr.Tab('模型列表'):
                        model_list = gr.Radio(models, label="", value=None)
                        quantization = gr.Radio([None,'4 bit','8 bit'],value=None,label='量化方式(不支持Windows)')
                        lora_list = gr.Radio(
                            [None]+loras, label="LoRA模型列表", value=None)
                        # use_deepspeed = gr.Checkbox(label='使用deepspeed')
                        with gr.Accordion('模型参数', open=True):
                            max_length = gr.Slider(
                                0, 4096, value=2048, step=256, label="Maximum length", interactive=True)
                            top_p = gr.Slider(0, 1, value=0.7, step=0.05,
                                            label="Top P", interactive=True)
                            temperature = gr.Slider(
                                0, 1, value=0.95, step=0.05, label="Temperature", interactive=True)

                    save2 = gr.Button("保存",variant="primary")
                    emptymodelBtn = gr.Button("清空")
            with gr.Column(scale=1):
                with gr.Accordion(label='语音', open=False):
                    with gr.Accordion(label='文字转语音', open=True):
                        # gen_type = gr.Radio(['本地','在线']+vits_models, label='合成方式',value='本地')
                        gen_type = gr.Radio(['本地']+vits_models, label='合成方式',value='本地')
                        lang = gr.Radio(['中文'],label='语言',value='中文')
                    with gr.Accordion(label='修改声线', open=True):
                        voice_style = gr.Radio(["默认"]+svcs, label='声线',value="默认")
                with gr.Accordion('选择设定',open=True):
                    play = gr.Radio(plays, label="*选择人设")
                    time_c = gr.Checkbox(label='时间感知')
                    emoticon = gr.Slider(0, 1, value=0, step=0.05, label="使用表情包概率:", interactive=True)
                    user_name = gr.Textbox( placeholder="Input...", lines=1, label="称呼我的名字（openai生效）")
                    with gr.Accordion('加载背景',open=False):
                        with gr.Tab('Embedding API'):
                            emb_api_list = gr.Radio([None]+embedding_api, label="",value=None)
                            with gr.Accordion('openai参数', open=True, visible=False) as emb_openai_params:
                                embedding_openai_api_key = gr.Textbox(
                                    lines=1, placeholder="Write Here...", label="openai_api_key:", type='password')
                                embedding_openai_port = gr.Textbox(
                                    lines=1, value='', label="VPN proxyPort:")
                            with gr.Accordion('azure openai参数', open=True, visible=False) as emb_azure_openai_params:
                                embedding_azure_api_key = gr.Textbox(
                                    lines=1, placeholder="Write Here...", label="azure_api_key:", type='password')
                                embedding_azure_endpoint = gr.Textbox(
                                    lines=1, value='', label="endpoint:(azure openai)")
                                embedding_azure_engine = gr.Textbox(
                                    lines=1, value='', label="engine:(azure openai)")
                        with gr.Tab('Embedding Model'):
                            emb_model_list = gr.Radio(embs, label="",value=None)
                        with gr.Accordion('选择 语言风格库'):
                            memory = gr.Radio([None]+backgrounds, label="",value=None)
                        with gr.Accordion('选择 背景记忆库'):
                            background = gr.Radio([None]+backgrounds, label="选择 背景库",value=None)
                    with gr.Accordion('联网搜索',open=False):
                        net = gr.Checkbox(label='联网搜索')
                        search = gr.Radio(['bing','google'], label="选择 搜索引擎",value='bing')
                        search_key=gr.Textbox(lines=1, placeholder="Write Here...",label="*search_key:",type= 'password')
                        result_len = gr.Slider(1, 20, value=4, step=1, label="搜索条数:", interactive=True)
                    save1 = gr.Button("保存设定",variant="primary")
                    emptyConfigBtn = gr.Button("清空设定")
        
        total_params = [openai_api_key, openai_port, azure_api_key, azure_endpoint, azure_engine,quantization,max_length,top_p,temperature]
        total_config_params = [embedding_openai_api_key, embedding_openai_port, embedding_azure_api_key, embedding_azure_endpoint, embedding_azure_engine]

        gen_type.change(handle_online_tts,inputs=gen_type,outputs= lang)
        Refresh.click(refresh_file, outputs=[play,memory,background,emb_model_list,model_list,lora_list])
        save1.click(load_config_params, [play,time_c,emoticon,user_name,net,search,search_key,result_len,memory,background,emb_api_list,emb_model_list]+total_config_params, [user_input,chatbot],show_progress=True)
        save2.click(load_llm_params, [api_list,model_list,lora_list]+total_params, [user_input,submitGroup],show_progress=True)
        submitBtn.click(chat_api.predict, [user_input,chatbot,gen_type,lang,voice_style,show_type], [user_input,chatbot,loud,audio],show_progress=False)
        emptyBtn.click(chat_api.reset_state, outputs=[user_input,chatbot], show_progress=True)
        emptymodelBtn.click(chat_api.clear, outputs=[chatbot,user_input,api_list,model_list,lora_list,submitGroup])
        emptyConfigBtn.click(chat_api.clear_config, outputs=[play,time_c,emoticon,user_name,memory,background,emb_api_list,emb_model_list,net,search_key], show_progress=True)

        show_type.change(switch_show_type,[show_type],outputs=[audio,canvas])

        model_list.change(model_select, inputs=[api_list, model_list, lora_list], outputs=[api_list, model_list, lora_list])
        api_list.change(api_select, inputs=[api_list, model_list, lora_list], outputs=[
                    openai_params, azure_openai_params,api_list, model_list, lora_list])
        
        emb_model_list.change(emb_model_select, inputs=[emb_api_list,emb_model_list], outputs=[emb_api_list,emb_model_list])
        emb_api_list.change(emb_api_select, inputs=[emb_api_list,emb_model_list], outputs=[emb_openai_params, emb_azure_openai_params,emb_api_list,emb_model_list])