import gradio as gr
import os
from utils.ui_utils import chat_base_model


def get_directories(path, unuse):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d not in unuse]


real_path = os.path.split(os.path.realpath(__file__))[0]
new_path = os.path.join(real_path, "..", "models", "LLM")
models = get_directories(new_path, ['runs', 'vector_store'])
new_path = os.path.join(real_path, "..", "models", "LoRA")
loras = get_directories(new_path, [])

chat_model = chat_base_model()

model_api = ['openai', 'azure openai', 'ernie bot',
             'ernie bot turbo', 'chatglm api', 'spark api','ali api']


def refresh_directories():
    new_path = os.path.join(real_path, "..", "models", "LLM")
    models = get_directories(new_path, ['runs', 'vector_store'])
    new_path = os.path.join(real_path, "..", "models", "LoRA")
    loras = get_directories(new_path, [])
    return gr.update(choices=models), gr.update(choices=[None]+loras)


def model_select(api, model, lora):
    if model == None:
        return gr.update(value=api), gr.update(value=model), gr.update(value=lora)
    else:
        return gr.update(value=None), gr.update(value=model), gr.update(value=lora)


def api_select(api, model, lora):
    if api == None:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=api), gr.update(value=model), gr.update(value=lora)
    elif api == 'openai':
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),gr.update(visible=False), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'azure openai':
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),gr.update(visible=False), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'ernie bot':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),gr.update(visible=False), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'ernie bot turbo':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'chatglm api':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),gr.update(visible=False), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'spark api':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'ali api':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),gr.update(value=api), gr.update(value=None), gr.update(value=None)
    else:
        pass

def load_params(api_list, model_list, lora_list, *args):
    params = {}
    params['prompt'] = args[0]
    if api_list is not None:
        if api_list == 'openai':
            params['name'] = 'openai'
            params['api_key'] = args[1]
            params['port'] = args[2]
        elif api_list == 'azure openai':
            params['name'] = 'azure openai'
            params['api_key'] = args[3]
            params['endpoint'] = args[4]
            params['engine'] = args[5]
        elif api_list == 'ernie bot':
            params['name'] = 'ernie bot'
            params['api_key'] = args[6]
            params['secret_key'] = args[7]
            params['temperature'] = args[8]
            params['top_p'] = args[9]
            params['penalty_score'] = args[10]
        elif api_list == 'ernie bot turbo':
            params['name'] = 'ernie bot turbo'
            params['api_key'] = args[11]
            params['secret_key'] = args[12]
        elif api_list == 'chatglm api':
            params['name'] = 'chatglm api'
            params['api_key'] = args[13]
            params['temperature'] = args[14]
            params['top_p'] = args[15]
            params['type'] = args[16]
        elif api_list == 'spark api':
            params['name'] = 'spark api'
            params['appid'] = args[17]
            params['api_key'] = args[18]
            params['secret_key'] = args[19]
            params['temperature'] = args[20]
            params['top_k'] = args[21]
            params['max_tokens'] = args[22]
        elif api_list == 'ali api':
            params['name'] = 'ali api'
            params['api_key'] = args[23]
            params['top_p'] = args[24]
            params['top_k'] = args[25]
            params['kuake_search'] = args[26]
        else:
            pass
        return chat_model.load_api_params(params)
    elif model_list is not None:
        params['name'] = model_list
        params['lora'] = lora_list
        params['quantization'] = args[27]
        params['max_length'] = args[28]
        params['top_p'] = args[29]
        params['temperature'] = args[30]
        params['use_deepspeed'] = args[31]
        return chat_model.load_model(params)
    raise gr.Error('请选择API或模型')


def chat_page():
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot()
            user_input = gr.Textbox(
                show_label=False, placeholder="Input...", lines=10,elem_id='chat-user-input')
            with gr.Accordion(label='', visible=False,elem_id='chat-submitGroup') as submitGroup:
                submitBtn = gr.Button("Submit", variant="primary",elem_id='chat-submitBtn')
                emptyBtn = gr.Button("Clear History")
        with gr.Column(scale=1):
            with gr.Accordion('', open=True):
                Refresh = gr.Button("刷新")
            with gr.Accordion('*选择模型', open=True):
                with gr.Tab('API列表'):
                    api_list = gr.Radio(model_api, label="", value=None,elem_id='chat_api_list')
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
                    with gr.Accordion('ernie bot参数', open=True, visible=False) as ernie_bot_params:
                        ernie_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                        ernie_secret_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*secret_key:", type='password')
                        ernie_temperature = gr.Slider(
                            0, 1, value=0.95, step=0.05, label="Temperature", interactive=True,elem_id='chat_ernie_temperature')
                        ernie_top_p = gr.Slider(
                            0, 1, value=0.8, step=0.05, label="Top P", interactive=True,elem_id='chat_ernie_top_p')
                        ernie_penalty_score = gr.Slider(
                            1, 2, value=1, step=0.05, label="Penalty Score", interactive=True,elem_id='chat_ernie_penalty_score')
                    with gr.Accordion('ernie bot turbo参数', open=True, visible=False) as ernie_bot_turbo_params:
                        ernie_turbo_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                        ernie_turbo_secret_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*secret_key:", type='password')
                    with gr.Accordion('chatglm参数', open=True, visible=False) as chatglm_params:
                        chatglm_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                        chatglm_temperature = gr.Slider(
                            0, 1, value=0.95, step=0.05, label="Temperature", interactive=True,elem_id='chat_chatglm_temperature')
                        chatglm_top_p = gr.Slider(
                            0, 1, value=0.8, step=0.05, label="Top P", interactive=True,elem_id='chat_chatglm_top_p')
                        chatglm_type = gr.Radio(
                            ['lite', 'std', 'pro'], label='模型类型', value='std')
                    with gr.Accordion('spark api参数', open=True, visible=False) as spark_params:
                        spark_appid = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*appid:", type='password')
                        spark_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                        spark_secret_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*secret_key:", type='password')
                        spark_temperature = gr.Slider(
                            0, 1, value=0.5, step=0.05, label="Temperature", interactive=True,elem_id='chat_spark_temperature')
                        spark_top_k = gr.Slider(
                            1, 6, value=4, step=1, label="Top K", interactive=True,elem_id='chat_spark_top_k')
                        spark_max_tokens = gr.Slider(
                            0, 4096, value=2048, step=256, label="Maximum tokens", interactive=True)
                    with gr.Accordion('ali api参数', open=True, visible=False) as ali_params:
                        ali_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                        ali_top_p = gr.Slider(
                            0, 1, value=0.8, step=0.05, label="Top p", interactive=True,elem_id='chat_ali_top_p')
                        ali_top_k = gr.Slider(
                            1, 100, value=100, step=1, label="Top k", interactive=True,elem_id='chat_ali_top_k')
                        ali_kuale_search = gr.Checkbox(label='联网搜索')
                with gr.Tab('模型列表'):
                    model_list = gr.Radio(models, label="", value=None)
                    # 量化方式(不支持Windows)
                    quantization = gr.Radio([None, '4 bit', '8 bit'],value=None, label="量化方式(不支持Windows)")
                    lora_list = gr.Radio(
                        [None]+loras, label="LoRA模型列表", value=None)
                    use_deepspeed = gr.Checkbox(label='使用deepspeed')
                    with gr.Accordion('模型参数', open=True):
                        max_length = gr.Slider(
                            0, 4096, value=2048, step=256, label="Maximum length", interactive=True,elem_id='chat_model_max_length')
                        top_p = gr.Slider(0, 1, value=0.7, step=0.05,
                                        label="Top P", interactive=True,elem_id='chat_model_top_p')
                        temperature = gr.Slider(
                            0, 1, value=0.95, step=0.05, label="Temperature", interactive=True,elem_id='chat_model_temperature')
                prompt = gr.Textbox(value="请可爱的风格回答下述问题。", lines=3,
                                    placeholder="Write Here...", label="提示词", interactive=True)

                save = gr.Button("确定", variant="primary")
                emptymodelBtn = gr.Button("清空")
            with gr.Accordion('联网搜索', open=False):
                net = gr.Checkbox(label='联网搜索')
                search = gr.Radio(['bing', 'google'],
                                  label="选择 搜索引擎", value='bing')
                search_key = gr.Textbox(
                    lines=1, placeholder="Write Here...", label="*联网key:", type='password')
                result_len = gr.Slider(
                    1, 20, value=3, step=1, label="搜索条数:", interactive=True)

        

    total_params = [prompt, openai_api_key, openai_port, azure_api_key, azure_endpoint, azure_engine, ernie_api_key, ernie_secret_key, ernie_temperature, ernie_top_p, ernie_penalty_score, ernie_turbo_api_key,
                    ernie_turbo_secret_key, chatglm_api_key, chatglm_temperature, chatglm_top_p, chatglm_type, spark_appid, spark_api_key, spark_secret_key, spark_temperature, spark_top_k, spark_max_tokens,ali_api_key,ali_top_p,ali_top_k,ali_kuale_search,quantization,max_length,top_p,temperature,use_deepspeed]
    history = gr.State([])

    Refresh.click(refresh_directories, outputs=[model_list,lora_list])
    # # save.click(chat_model.load, [params_state,models0,lora0,max_length,top_p,temperature,openai_api_key3,port3,use_deepspeed,endpoint,engine,prompt,ernie_api_key,ernie_secret_key,ernie_temperature,ernie_top_p,ernie_penalty_score,chatglm_api_key,chatglm_temperature,chatglm_top_p,chatglm_type,spark_appid,spark_api_key,spark_secret_key,spark_temperature,spark_top_k,spark_max_length], [user_input,submitGroup],show_progress=True)
    save.click(load_params, [api_list, model_list, lora_list]+total_params,
               outputs=[user_input, submitGroup], show_progress=True)
    submitBtn.click(chat_model.predict, [user_input, chatbot, history,net,search,search_key,result_len,prompt], [chatbot, history,user_input],show_progress=True)

    emptyBtn.click(chat_model.reset_state, outputs=[chatbot,history,user_input])
    emptymodelBtn.click(chat_model.clear, outputs=[chatbot,history,user_input,api_list,model_list,lora_list,submitGroup])

    model_list.change(model_select, inputs=[api_list, model_list, lora_list], outputs=[api_list, model_list, lora_list])
    api_list.change(api_select, inputs=[api_list, model_list, lora_list], outputs=[
                    openai_params, azure_openai_params, ernie_bot_params, ernie_bot_turbo_params, chatglm_params, spark_params, ali_params,api_list, model_list, lora_list])
