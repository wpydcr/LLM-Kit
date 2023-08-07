import gradio as gr
import os
from utils.ui_utils import chat_base_model
from utils.parallel_api import Parallel_api, ParallelLocalModel


def get_directories(path, unuse):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d not in unuse]


real_path = os.path.split(os.path.realpath(__file__))[0]
new_path = os.path.join(real_path, "..", "models", "LLM")
models = get_directories(new_path, ['runs', 'vector_store'])
new_path = os.path.join(real_path, "..", "models", "LoRA")
loras = get_directories(new_path, [])
new_path = os.path.join(real_path, "..", "models", "Embedding")
embs = get_directories(new_path, [])
new_path = os.path.join(real_path, "..", "data", "documents")
docs = get_directories(new_path, [])

chat_model = chat_base_model()
parallel_api = Parallel_api()

model_api = ['openai', 'azure openai', 'ernie bot',
             'ernie bot turbo', 'chatglm api', 'spark api', 'ali api']

embedding_api = ['openai', 'azure openai']
parallel_local_model = ParallelLocalModel()

def refresh_directories():
    new_path = os.path.join(real_path, "..", "models", "LLM")
    models = get_directories(new_path, ['runs', 'vector_store'])
    new_path = os.path.join(real_path, "..", "models", "LoRA")
    loras = get_directories(new_path, [])
    return gr.update(choices=models), gr.update(choices=[None]+loras)


def refresh_embedding_directories():
    new_path = os.path.join(real_path, "..", "models", "Embedding")
    embs = get_directories(new_path, [])
    new_path = os.path.join(real_path, "..", "data", "documents")
    docs = get_directories(new_path, [])
    return gr.update(choices=embs), gr.update(choices=docs)


def model_select(api, model, lora):
    if model == None:
        return gr.update(value=api), gr.update(value=model), gr.update(value=lora)
    else:
        return gr.update(value=None), gr.update(value=model), gr.update(value=lora)


def api_page_clear():
    parallel_api.clear()
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value='', visible=False), gr.update(visible=False), gr.update(value=None), gr.update(value=''), gr.update(value=False), gr.update(value=None), gr.update(value=None), gr.update(value=None)


def api_select(api, model, lora):
    if api == None:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=api), gr.update(value=model), gr.update(value=lora)
    elif api == 'openai':
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'azure openai':
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'ernie bot':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'ernie bot turbo':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'chatglm api':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'spark api':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    elif api == 'ali api':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=api), gr.update(value=None), gr.update(value=None)
    else:
        pass


def load_params(api_list, model_list, lora_list, *args):
    apis = ['openai','azure openai','ernie bot', 'ernie bot turbo','chatglm api','spark api','ali api']
    params = {}
    params['prompt'] = args[0]
    if api_list in apis:
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


def show_api_params_add_api(api_list):
    res = []
    if 'openai' in api_list:
        res.append(gr.update(visible=True))
    else:
        res.append(gr.update(visible=False))
    if 'azure openai' in api_list:
        res.append(gr.update(visible=True))
    else:
        res.append(gr.update(visible=False))
    if 'ernie bot' in api_list:
        res.append(gr.update(visible=True))
    else:
        res.append(gr.update(visible=False))
    if 'ernie bot turbo' in api_list:
        res.append(gr.update(visible=True))
    else:
        res.append(gr.update(visible=False))
    if 'chatglm api' in api_list:
        res.append(gr.update(visible=True))
    else:
        res.append(gr.update(visible=False))
    if 'spark api' in api_list:
        res.append(gr.update(visible=True))
    else:
        res.append(gr.update(visible=False))
    if 'ali api' in api_list:
        res.append(gr.update(visible=True))
    else:
        res.append(gr.update(visible=False))
    return res[0], res[1], res[2], res[3], res[4], res[5], res[6], gr.update(value=' | '.join(api_list))


def show_knowledge(use_knowledge):
    if use_knowledge:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def emb_model_select(emb_api, emb_model):
    if emb_model is None:
        return gr.update(value=emb_api), gr.update(value=emb_model)
    else:
        return gr.update(value=None), gr.update(value=emb_model)


def emb_api_select(emb_api, emb_model):
    if emb_api is None:
        return gr.update(visible=False), gr.update(visible=False), gr.update(value=emb_api), gr.update(value=emb_model)
    elif emb_api == 'openai':
        return gr.update(visible=True), gr.update(visible=False), gr.update(value=emb_api), gr.update(value=None)
    elif emb_api == 'azure openai':
        return gr.update(visible=False), gr.update(visible=True), gr.update(value=emb_api), gr.update(value=None)
    else:
        pass


def load_api_page_params(api_api_list, api_emb_api_list, api_emb_model_list, api_doc1, api_k, api_score_threshold, api_chunk_size, api_chunk_conent, api_net, api_search, api_search_key, api_result_len, api_prompt, use_knowledge, *args):
    if api_api_list == []:
        raise gr.Error('请选择API')
    if use_knowledge and api_emb_api_list is None and api_emb_model_list is None:
        raise gr.Error('请选择嵌入模型或API')
    if use_knowledge and api_doc1 is None:
        raise gr.Error('请选择文档')
    if api_net and api_search_key == '':
        raise gr.Error('请输入联网key')
    params = {}
    returns = []
    params['api_list'] = []
    if 'openai' in api_api_list:
        params['api_list'].append('openai')
        params['openai_api_key'] = args[0]
        params['openai_port'] = args[1]
        returns.append(gr.update(visible=True,value=None))
        returns.append(gr.update(visible=True if use_knowledge else False,value=None))
    else:
        returns.append(gr.update(visible=False))
        returns.append(gr.update(visible=False))
    if 'azure openai' in api_api_list:
        params['api_list'].append('azure openai')
        params['azure_api_key'] = args[2]
        params['azure_endpoint'] = args[3]
        params['azure_engine'] = args[4]
        returns.append(gr.update(visible=True,value=None))
        returns.append(gr.update(visible=True if use_knowledge else False,value=None))
    else:
        returns.append(gr.update(visible=False))
        returns.append(gr.update(visible=False))
    if 'ernie bot' in api_api_list:
        params['api_list'].append('ernie bot')
        params['ernie_api_key'] = args[5]
        params['ernie_secret_key'] = args[6]
        params['ernie_temperature'] = args[7]
        params['ernie_top_p'] = args[8]
        params['ernie_penalty_score'] = args[9]
        returns.append(gr.update(visible=True,value=None))
        returns.append(gr.update(visible=True if use_knowledge else False,value=None))
    else:
        returns.append(gr.update(visible=False))
        returns.append(gr.update(visible=False))
    if 'ernie bot turbo' in api_api_list:
        params['api_list'].append('ernie bot turbo')
        params['ernie_turbo_api_key'] = args[10]
        params['ernie_turbo_secret_key'] = args[11]
        returns.append(gr.update(visible=True,value=None))
        returns.append(gr.update(visible=True if use_knowledge else False,value=None))
    else:
        returns.append(gr.update(visible=False))
        returns.append(gr.update(visible=False))
    if 'chatglm api' in api_api_list:
        params['api_list'].append('chatglm api')
        params['chatglm_api_key'] = args[12]
        params['chatglm_temperature'] = args[13]
        params['chatglm_top_p'] = args[14]
        params['chatglm_type'] = args[15]
        returns.append(gr.update(visible=True,value=None))
        returns.append(gr.update(visible=True if use_knowledge else False,value=None))
    else:
        returns.append(gr.update(visible=False))
        returns.append(gr.update(visible=False))
    if 'spark api' in api_api_list:
        params['api_list'].append('spark api')
        params['spark_appid'] = args[16]
        params['spark_api_key'] = args[17]
        params['spark_secret_key'] = args[18]
        params['spark_temperature'] = args[19]
        params['spark_top_k'] = args[20]
        params['spark_max_tokens'] = args[21]
        returns.append(gr.update(visible=True,value=None))
        returns.append(gr.update(visible=True if use_knowledge else False,value=None))
    else:
        returns.append(gr.update(visible=False))
        returns.append(gr.update(visible=False))
    if 'ali api' in api_api_list:
        params['api_list'].append('ali api')
        params['ali_api_key'] = args[22]
        params['ali_top_p'] = args[23]
        params['ali_top_k'] = args[24]
        params['ali_kuake_search'] = args[25]
        returns.append(gr.update(visible=True,value=None))
        returns.append(gr.update(visible=True if use_knowledge else False,value=None))
    else:
        returns.append(gr.update(visible=False))
        returns.append(gr.update(visible=False))
    if use_knowledge:
        params['use_knowledge'] = True
        if api_emb_api_list is not None:
            if 'openai' == api_emb_api_list:
                params['emb_name'] = 'openai'
                params['emb_api_key'] = args[26]
                params['emb_port'] = args[27]
            elif 'azure openai' == api_emb_api_list:
                params['emb_api'] = 'azure openai'
                params['emb_api_key'] = args[28]
                params['emb_endpoint'] = args[29]
                params['emb_engine'] = args[30]
        elif api_emb_model_list is not None:
            params['emb_name'] = api_emb_model_list
        else:
            pass
        params['doc'] = api_doc1
        params['k'] = api_k
        params['score_threshold'] = api_score_threshold
        params['chunk_size'] = api_chunk_size
        params['chunk_conent'] = api_chunk_conent
    else:
        params['use_knowledge'] = False
    if api_net:
        params['net'] = True
        params['search'] = api_search
        params['search_key'] = api_search_key
        params['result_len'] = api_result_len
    else:
        params['net'] = False
    params['prompt'] = api_prompt
    if parallel_api.setv(params):
        return returns[0], returns[1], returns[2], returns[3], returns[4], returns[5], returns[6], returns[7], returns[8], returns[9], returns[10], returns[11], returns[12], returns[13], gr.update(visible=True,interactive=True), gr.update(visible=True)


def clear_history():
    parallel_api.clear_history()
    return gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None)


def chat_page(localizer):

    with gr.Tab(localizer('聊天')):
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot()
                user_input = gr.Textbox(
                    show_label=False, placeholder="Input...", lines=10, elem_id='chat-user-input')
                with gr.Accordion(label='', visible=False, elem_id='chat-submitGroup') as submitGroup:
                    submitBtn = gr.Button(
                        localizer("提交"), variant="primary", elem_id='chat-submitBtn')
                    emptyBtn = gr.Button(localizer("清除"))
            with gr.Column(scale=1):
                with gr.Accordion('', open=True):
                    Refresh = gr.Button(localizer("刷新"))
                with gr.Accordion(localizer('选择模型'), open=True):
                    with gr.Tab(localizer('API列表')):
                        api_list = gr.Radio(
                            model_api, show_label=False, value=None, elem_id='chat_api_list')
                        with gr.Accordion(localizer("openai参数"), open=True, visible=False) as openai_params:
                            openai_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*openai_api_key:", type='password')
                            openai_port = gr.Textbox(
                                lines=1, value='', label="*VPN proxyPort:")
                        with gr.Accordion(localizer("azure openai参数"), open=True, visible=False) as azure_openai_params:
                            azure_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*azure_api_key:", type='password')
                            azure_endpoint = gr.Textbox(
                                lines=1, value='', label="*endpoint:(azure openai)")
                            azure_engine = gr.Textbox(
                                lines=1, value='', label="*engine:(azure openai)")
                        with gr.Accordion(localizer("ernie bot参数"), open=True, visible=False) as ernie_bot_params:
                            ernie_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                            ernie_secret_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*secret_key:", type='password')
                            ernie_temperature = gr.Slider(
                                0, 1, value=0.95, step=0.05, label="Temperature", interactive=True, elem_id='chat_ernie_temperature')
                            ernie_top_p = gr.Slider(
                                0, 1, value=0.8, step=0.05, label="Top P", interactive=True, elem_id='chat_ernie_top_p')
                            ernie_penalty_score = gr.Slider(
                                1, 2, value=1, step=0.05, label="Penalty Score", interactive=True, elem_id='chat_ernie_penalty_score')
                        with gr.Accordion(localizer("ernie bot turbo参数"), open=True, visible=False) as ernie_bot_turbo_params:
                            ernie_turbo_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                            ernie_turbo_secret_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*secret_key:", type='password')
                        with gr.Accordion(localizer("chatglm参数"), open=True, visible=False) as chatglm_params:
                            chatglm_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                            chatglm_temperature = gr.Slider(
                                0, 1, value=0.95, step=0.05, label="Temperature", interactive=True, elem_id='chat_chatglm_temperature')
                            chatglm_top_p = gr.Slider(
                                0, 1, value=0.8, step=0.05, label="Top P", interactive=True, elem_id='chat_chatglm_top_p')
                            chatglm_type = gr.Radio(
                                ['lite', 'std', 'pro'], label=localizer("模型类型"), value='std')
                        with gr.Accordion(localizer("spark api参数"), open=True, visible=False) as spark_params:
                            spark_appid = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*appid:", type='password')
                            spark_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                            spark_secret_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*secret_key:", type='password')
                            spark_temperature = gr.Slider(
                                0, 1, value=0.5, step=0.05, label="Temperature", interactive=True, elem_id='chat_spark_temperature')
                            spark_top_k = gr.Slider(
                                1, 6, value=4, step=1, label="Top K", interactive=True, elem_id='chat_spark_top_k')
                            spark_max_tokens = gr.Slider(
                                0, 4096, value=2048, step=256, label="Maximum tokens", interactive=True)
                        with gr.Accordion(localizer("ali api参数"), open=True, visible=False) as ali_params:
                            ali_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                            ali_top_p = gr.Slider(
                                0, 1, value=0.8, step=0.05, label="Top p", interactive=True, elem_id='chat_ali_top_p')
                            ali_top_k = gr.Slider(
                                1, 100, value=100, step=1, label="Top k", interactive=True, elem_id='chat_ali_top_k')
                            ali_kuake_search = gr.Checkbox(label=localizer("联网搜索"))
                    with gr.Tab(localizer("选择模型")):
                        model_list = gr.Radio(models, show_label=False, value=None)
                        # 量化方式(不支持Windows)
                        quantization = gr.Radio(
                            [None, '4 bit', '8 bit'], value=None, label=localizer("量化方式(不支持Windows)"))
                        lora_list = gr.Radio(
                            [None]+loras, label=localizer("LoRA模型列表"), value=None)
                        use_deepspeed = gr.Checkbox(label=localizer("使用deepspeed"))
                        with gr.Accordion(localizer("模型参数"), open=True):
                            max_length = gr.Slider(
                                0, 4096, value=2048, step=256, label="Maximum length", interactive=True, elem_id=localizer("模型参数"))
                            top_p = gr.Slider(0, 1, value=0.7, step=0.05,
                                              label="Top P", interactive=True, elem_id='chat_model_top_p')
                            temperature = gr.Slider(
                                0, 1, value=0.95, step=0.05, label="Temperature", interactive=True, elem_id='chat_model_temperature')
                    stream = gr.Checkbox(label=localizer('流式输出'))
                    prompt = gr.Textbox(value=localizer("请用可爱的风格回答下述问题。"), lines=3,
                                        placeholder="Write Here...", label=localizer("提示词"), interactive=True)

                    save = gr.Button(localizer("确定"), variant="primary")
                    emptymodelBtn = gr.Button(localizer("清空"))

                with gr.Accordion(localizer("联网搜索"), open=False):
                    net = gr.Checkbox(label=localizer("联网搜索"))
                    search = gr.Radio(['bing', 'google'],
                                      label=localizer("选择搜索引擎"), value='bing')
                    search_key = gr.Textbox(
                        lines=1, placeholder="Write Here...", label=localizer("*联网key:"), type='password')
                    result_len = gr.Slider(
                        1, 20, value=3, step=1, label=localizer("搜索条数:"), interactive=True)
    with gr.Tab(localizer("API并行调用")):
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    api_openai_chatbot = gr.Chatbot(
                        visible=False, label='openai', height=200)
                    api_openai_knowledge_chatbot = gr.Chatbot(
                        visible=False, label=localizer("openai+知识库"), height=200)
                with gr.Row():
                    api_azure_chatbot = gr.Chatbot(
                        visible=False, label='azure openai', height=200)
                    api_azure_knowledge_chatbot = gr.Chatbot(
                        visible=False, label=localizer("azure openai+知识库"), height=200)
                with gr.Row():
                    api_ernie_chatbot = gr.Chatbot(
                        visible=False, label='ernie bot', height=200)
                    api_ernie_knowledge_chatbot = gr.Chatbot(
                        visible=False, label=localizer("ernie bot+知识库"), height=200)
                with gr.Row():
                    api_ernie_turbo_chatbot = gr.Chatbot(
                        visible=False, label='ernie bot turbo', height=200)
                    api_ernie_turbo_knowledge_chatbot = gr.Chatbot(
                        visible=False, label=localizer("ernie bot turbo+知识库"), height=200)
                with gr.Row():
                    api_chatglm_chatbot = gr.Chatbot(
                        visible=False, label='chatglm api', height=200)
                    api_chatglm_knowledge_chatbot = gr.Chatbot(
                        visible=False, label=localizer("chatglm api+知识库"), height=200)
                with gr.Row():
                    api_spark_chatbot = gr.Chatbot(
                        visible=False, label='spark api', height=200)
                    api_spark_knowledge_chatbot = gr.Chatbot(
                        visible=False, label=localizer("spark api+知识库"), height=200)
                with gr.Row():
                    api_ali_chatbot = gr.Chatbot(
                        visible=False, label='ali api', height=200)
                    api_ali_knowledge_chatbot = gr.Chatbot(
                        visible=False, label=localizer("ali api+知识库"), height=200)
                api_user_input = gr.Textbox(
                    show_label=False, placeholder="Input...", lines=10, elem_id='chat-api-user-input', visible=False,interactive=True)
                with gr.Accordion(label='', visible=False, elem_id='chat-api-submitGroup') as api_submitGroup:
                    api_submitBtn = gr.Button(
                        localizer("提交"), variant="primary", elem_id='chat-api-submitBtn')
                    api_emptyBtn = gr.Button(localizer("清除记录"))
            with gr.Column(scale=1):
                with gr.Accordion(localizer("选择API"), open=True):
                    api_api_list = gr.CheckboxGroup(
                        model_api, show_label=False, value=None, elem_id='chat_api_api_list')
                    with gr.Accordion(localizer("openai参数"), open=False, visible=False) as api_openai_params:
                        api_openai_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*openai_api_key:", type='password')
                        api_openai_port = gr.Textbox(
                            lines=1, value='', label="*VPN proxyPort:")
                    with gr.Accordion(localizer("azure openai参数"), open=False, visible=False) as api_azure_openai_params:
                        api_azure_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*azure_api_key:", type='password')
                        api_azure_endpoint = gr.Textbox(
                            lines=1, value='', label="*endpoint:(azure openai)")
                        api_azure_engine = gr.Textbox(
                            lines=1, value='', label="*engine:(azure openai)")
                    with gr.Accordion(localizer("ernie bot参数"), open=False, visible=False) as api_ernie_bot_params:
                        api_ernie_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                        api_ernie_secret_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*secret_key:", type='password')
                        api_ernie_temperature = gr.Slider(
                            0, 1, value=0.95, step=0.05, label="Temperature", interactive=True, elem_id='chat_ernie_temperature')
                        api_ernie_top_p = gr.Slider(
                            0, 1, value=0.8, step=0.05, label="Top P", interactive=True, elem_id='chat_ernie_top_p')
                        api_ernie_penalty_score = gr.Slider(
                            1, 2, value=1, step=0.05, label="Penalty Score", interactive=True, elem_id='chat_ernie_penalty_score')
                    with gr.Accordion(localizer("ernie bot turbo参数"), open=False, visible=False) as api_ernie_bot_turbo_params:
                        api_ernie_turbo_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                        api_ernie_turbo_secret_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*secret_key:", type='password')
                    with gr.Accordion(localizer("chatglm参数"), open=False, visible=False) as api_chatglm_params:
                        api_chatglm_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                        api_chatglm_temperature = gr.Slider(
                            0, 1, value=0.95, step=0.05, label="Temperature", interactive=True, elem_id='chat_chatglm_temperature')
                        api_chatglm_top_p = gr.Slider(
                            0, 1, value=0.8, step=0.05, label="Top P", interactive=True, elem_id='chat_chatglm_top_p')
                        api_chatglm_type = gr.Radio(
                            ['lite', 'std', 'pro'], label=localizer("模型类型"), value='std')
                    with gr.Accordion(localizer("spark api参数"), open=False, visible=False) as api_spark_params:
                        api_spark_appid = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*appid:", type='password')
                        api_spark_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                        api_spark_secret_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*secret_key:", type='password')
                        api_spark_temperature = gr.Slider(
                            0, 1, value=0.5, step=0.05, label="Temperature", interactive=True, elem_id='chat_spark_temperature')
                        api_spark_top_k = gr.Slider(
                            1, 6, value=4, step=1, label="Top K", interactive=True, elem_id='chat_spark_top_k')
                        api_spark_max_tokens = gr.Slider(
                            0, 4096, value=2048, step=256, label="Maximum tokens", interactive=True)
                    with gr.Accordion(localizer("ali api参数"), open=False, visible=False) as api_ali_params:
                        api_ali_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*api_key:", type='password')
                        api_ali_top_p = gr.Slider(
                            0, 1, value=0.8, step=0.05, label="Top p", interactive=True, elem_id='chat_ali_top_p')
                        api_ali_top_k = gr.Slider(
                            1, 100, value=100, step=1, label="Top k", interactive=True, elem_id='chat_ali_top_k')
                        api_ali_kuake_search = gr.Checkbox(label=localizer('联网搜索'))

                    api_selected_api = gr.Textbox(
                        label=localizer("已选择的模型"), lines=1, value=None, interactive=False)
                    api_stream = gr.Checkbox(label=localizer('流式输出'))
                    api_prompt = gr.Textbox(value=localizer("请用可爱的风格回答下述问题。"), lines=3,
                                            placeholder="Write Here...", label=localizer("提示词"), interactive=True)
                    use_knowledge = gr.Checkbox(label=localizer('使用知识库'))
                with gr.Accordion(localizer('本地知识库'), open=True, visible=False) as api_local_knowledge:
                    api_emb_refresh = gr.Button(localizer("刷新"))
                    with gr.Tab('Embedding API'):
                        api_emb_api_list = gr.Radio(
                            embedding_api, show_label=False, value=None)
                        with gr.Accordion(localizer('openai参数'), open=True, visible=False) as api_emb_openai_params:
                            api_embedding_openai_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*openai_api_key:", type='password')
                            api_embedding_openai_port = gr.Textbox(
                                lines=1, value='', label="*VPN proxyPort:")
                        with gr.Accordion(localizer('azure openai参数'), open=True, visible=False) as api_emb_azure_openai_params:
                            api_embedding_azure_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*azure_api_key:", type='password')
                            api_embedding_azure_endpoint = gr.Textbox(
                                lines=1, value='', label="*endpoint:(azure openai)")
                            api_embedding_azure_engine = gr.Textbox(
                                lines=1, value='', label="*engine:(azure openai)")
                    with gr.Tab(localizer(localizer('嵌入式模型'))):
                        api_emb_model_list = gr.Radio(
                            embs, show_label=False, value=None)
                    with gr.Accordion(localizer('*选择向量知识库'), open=False):
                        api_doc1 = gr.Radio(docs, show_label=False, value=None)
                    api_k = gr.Slider(
                        1, 20, value=3, step=1, label=localizer("使用前几条相关文本"), interactive=True)
                    api_score_threshold = gr.Slider(

                        0, 1100, value=500, step=1, label=localizer("相似度阈值(0不生效)"), interactive=True)
                    api_chunk_size = gr.Slider(
                        1, 2048, value=250, step=1, label=localizer("每条文本长度"), interactive=True)
                    api_chunk_conent = gr.Checkbox(label=localizer('相似文本是否启用上下文查询'))
                with gr.Accordion(label=''):
                    api_save = gr.Button(localizer("确定"), variant="primary")
                    api_emptymodelBtn = gr.Button(localizer("清空"))
                with gr.Accordion(localizer('联网搜索'), open=False):
                    api_net = gr.Checkbox(label=localizer('联网搜索'))
                    api_search = gr.Radio(['bing', 'google'],
                                          label=localizer("选择搜索引擎"), value='bing')
                    api_search_key = gr.Textbox(
                        lines=1, placeholder="Write Here...", label=localizer("*联网key:"), type='password')
                    api_result_len = gr.Slider(
                        1, 20, value=3, step=1, label=localizer("搜索条数:"), interactive=True)

    with gr.Tab(localizer("本地模型数据库并行调用")) as dual_local:
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column():
                        local_chatbot1 = gr.Chatbot(visible=False,label="Chatbot1")
                    with gr.Column():
                        local_chatbot2 = gr.Chatbot(visible=False,label="Chatbot2")
                with gr.Row():
                    parallel_local_model_textinput = gr.Textbox(
                    show_label=False, placeholder="Input...", lines=10, elem_id='chat-model-user-input')
                with gr.Accordion(label='', visible=False, elem_id='chat-model-submitGroup') as parallel_local_model_submitGroup:
                    parallel_local_model_submitBtn = gr.Button(
                        localizer("submit"), variant="primary", elem_id='chat-model-submitBtn')
                    parallel_local_model_emptyBtn = gr.Button(localizer("Clear History"))
            with gr.Column(scale=1):
                with gr.Accordion(localizer("载入的向量数据库"),open=True):
                    status = gr.Dataframe(headers=["Chatbot1", "Chatbot2"],
                                        datatype=["str", "str"],
                                        show_label=False,row_count = (1,"fixed"),col_count=(2,"fixed"))

                with gr.Accordion(localizer("模型设置")):
                    parallel_local_model_model_list = gr.Radio(models, show_label=False, value=None)
                    # 量化方式(不支持Windows)
                    parallel_local_model_quantization = gr.Radio(
                        [None, '4 bit', '8 bit'], value=None, label=localizer("量化方式(不支持Windows)"))
                    parallel_local_model_lora_list = gr.Radio(
                        [None] + loras, label=localizer("LoRA模型列表"), value=None)
                    parallel_local_model_use_deepspeed = gr.Checkbox(label=localizer('使用deepspeed'))
                    with gr.Accordion(localizer('模型参数'), open=False):
                        parallel_local_model_max_length = gr.Slider(
                            0, 4096, value=2048, step=256, label="Maximum length", interactive=True,
                            elem_id='chat_model_max_length')
                        parallel_local_model_top_p = gr.Slider(0, 1, value=0.7, step=0.05,
                                                               label="Top P", interactive=True)
                        parallel_local_model_temperature = gr.Slider(
                            0, 1, value=0.95, step=0.05, label="Temperature", interactive=True,
                            elem_id='chat_model_temperature')
                    parallel_local_model_prompt = gr.Textbox(value=localizer("请用可爱的风格回答下述问题。"), lines=3,
                                        placeholder="Write Here...", label=localizer("提示词"), interactive=True)
                    parallel_local_model_save = gr.Button(localizer("确定"), variant="primary")
                    parallel_local_model_emptymodelBtn = gr.Button(localizer("清空"))
                with gr.Accordion(localizer(localizer("使用知识库")), open=False):
                    switch_chatbot = gr.Radio(["chatbot1","chatbot2"], label=localizer("清空"), value=None)
                    with gr.Tab(localizer("Embedding API")):
                        emb_api_list = gr.Radio(embedding_api, show_label=False, value=None)
                        with gr.Accordion(localizer('openai参数'), open=True, visible=False) as emb_openai_params:
                            embedding_openai_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*openai_api_key:", type='password')
                            embedding_openai_port = gr.Textbox(
                                lines=1, value='', label="*VPN proxyPort:")
                        with gr.Accordion(localizer('azure openai参数'), open=True, visible=False) as emb_azure_openai_params:
                            embedding_azure_api_key = gr.Textbox(
                                lines=1, placeholder="Write Here...", label="*azure_api_key:", type='password')
                            embedding_azure_endpoint = gr.Textbox(
                                lines=1, value='', label="*endpoint:(azure openai)")
                            embedding_azure_engine = gr.Textbox(
                                lines=1, value='', label="*engine:(azure openai)")
                    with gr.Tab(localizer('嵌入式模型')):
                        emb_model_list = gr.Radio(embs, show_label=False, value=None)
                    with gr.Accordion(localizer('*选择向量知识库'), open=True):
                        doc1 = gr.Radio(docs, show_label=False, value=None)
                    k = gr.Slider(1, 20, value=3, step=1, label=localizer("使用前几条相关文本"), interactive=True)
                    score_threshold = gr.Slider(0, 1100, value=500, step=1, label=localizer("相似度阈值(0不生效)"),
                                                interactive=True)
                    chunk_size = gr.Slider(1, 2048, value=250, step=1, label=localizer("每条文本长度"), interactive=True)
                    chunk_conent = gr.Checkbox(label=localizer('相似文本是否启用上下文查询'))
                    save0 = gr.Button(localizer("确定知识库"), variant="primary")
                    emptymodelBtn0 = gr.Button(localizer("取消知识库"))
                with gr.Accordion(localizer('联网搜索'), open=False):
                    parallel_local_model_net = gr.Checkbox(label=localizer('联网搜索'))
                    parallel_local_model_search = gr.Radio(['bing', 'google'],
                                      label=localizer("选择搜索引擎"), value='bing')
                    parallel_local_model_search_key = gr.Textbox(
                        lines=1, placeholder="Write Here...", label=localizer("*联网key:"), type='password')
                    parallel_local_model_result_len = gr.Slider(
                        1, 20, value=3, step=1, label=localizer("搜索条数:"), interactive=True)

    dual_local.select(ParallelLocalModel.handle_local_model_selected, outputs=[local_chatbot1, local_chatbot2])
    embedding_total_params = [embedding_openai_api_key, embedding_openai_port, embedding_azure_api_key, embedding_azure_endpoint, embedding_azure_engine]

    save0.click(parallel_local_model.load_embedding_params, [status,switch_chatbot,doc1, k, score_threshold, chunk_size, chunk_conent, emb_api_list,
                                        emb_model_list] + embedding_total_params,[status], show_progress=True)
    emptymodelBtn0.click(parallel_local_model.clear, inputs=[switch_chatbot,status],outputs=[doc1, emb_api_list, emb_model_list,status])

    emb_api_list.change(emb_api_select, inputs=[emb_api_list, emb_model_list],
                        outputs=[emb_openai_params, emb_azure_openai_params, emb_api_list, emb_model_list])

    total_params = [prompt, openai_api_key, openai_port, azure_api_key, azure_endpoint, azure_engine, ernie_api_key, ernie_secret_key, ernie_temperature, ernie_top_p, ernie_penalty_score, ernie_turbo_api_key,
                    ernie_turbo_secret_key, chatglm_api_key, chatglm_temperature, chatglm_top_p, chatglm_type, spark_appid, spark_api_key, spark_secret_key, spark_temperature, spark_top_k, spark_max_tokens, ali_api_key, ali_top_p, ali_top_k, ali_kuake_search, quantization, max_length, top_p, temperature, use_deepspeed]

    history = gr.State([])

    parallel_history1 = gr.State([])
    parallel_history2 = gr.State([])
    parallel_local_model_emptymodelBtn.click(chat_model.clears, outputs=[local_chatbot1,local_chatbot2,parallel_history1,\
                                                                        parallel_history2,parallel_local_model_textinput,\
                                                                        parallel_local_model_model_list,parallel_local_model_lora_list,\
                                                                        parallel_local_model_submitGroup])

    parallel_local_model_submitBtn.click(chat_model.parallel_predict, [parallel_local_model_textinput, local_chatbot1,local_chatbot2,\
                                                                    parallel_history1,parallel_history2, parallel_local_model_net,\
                                                                    parallel_local_model_search, parallel_local_model_search_key,\
                                                                    parallel_local_model_result_len, parallel_local_model_prompt,status],
                                                                    [local_chatbot1,local_chatbot2, parallel_history1,parallel_history2,\
                                                                            parallel_local_model_textinput], show_progress=True)

    parallel_local_model_emptyBtn.click(chat_model.reset_states, outputs=[local_chatbot1,local_chatbot2, parallel_history1,parallel_history2, user_input])



    Refresh.click(refresh_directories, outputs=[model_list, lora_list])
    save.click(load_params, [api_list, model_list, lora_list]+total_params,
               outputs=[user_input, submitGroup], show_progress=True)

    parallel_local_model_total_params = [parallel_local_model_prompt, openai_api_key, openai_port, azure_api_key, azure_endpoint, azure_engine, ernie_api_key, ernie_secret_key, ernie_temperature, ernie_top_p, ernie_penalty_score, ernie_turbo_api_key,
                    ernie_turbo_secret_key, chatglm_api_key, chatglm_temperature, chatglm_top_p, chatglm_type, spark_appid, spark_api_key, spark_secret_key, spark_temperature, spark_top_k, spark_max_tokens, ali_api_key, ali_top_p, ali_top_k, ali_kuake_search, parallel_local_model_quantization, parallel_local_model_max_length, parallel_local_model_top_p, parallel_local_model_temperature, parallel_local_model_use_deepspeed]



    parallel_local_model_save.click(load_params, [parallel_local_model_model_list, parallel_local_model_model_list, parallel_local_model_lora_list]+parallel_local_model_total_params,
               outputs=[parallel_local_model_textinput, parallel_local_model_submitGroup], show_progress=True)

    submitBtn.click(chat_model.predict, [user_input, chatbot, history, stream, net, search, search_key, result_len, prompt], [
                    chatbot, history, user_input], show_progress=True)

    emptyBtn.click(chat_model.reset_state, outputs=[
                   chatbot, history, user_input])
    emptymodelBtn.click(chat_model.clear, outputs=[
                        chatbot, history, user_input, api_list, model_list, lora_list, submitGroup])

    model_list.change(model_select, inputs=[api_list, model_list, lora_list], outputs=[
                      api_list, model_list, lora_list])
    api_list.change(api_select, inputs=[api_list, model_list, lora_list], outputs=[
                    openai_params, azure_openai_params, ernie_bot_params, ernie_bot_turbo_params, chatglm_params, spark_params, ali_params, api_list, model_list, lora_list])

    # API并行调用页面
    api_total_params = [api_openai_api_key, api_openai_port, api_azure_api_key, api_azure_endpoint, api_azure_engine, api_ernie_api_key, api_ernie_secret_key, api_ernie_temperature, api_ernie_top_p, api_ernie_penalty_score, api_ernie_turbo_api_key,
                        api_ernie_turbo_secret_key, api_chatglm_api_key, api_chatglm_temperature, api_chatglm_top_p, api_chatglm_type, api_spark_appid, api_spark_api_key, api_spark_secret_key, api_spark_temperature, api_spark_top_k, api_spark_max_tokens, api_ali_api_key, api_ali_top_p, api_ali_top_k, api_ali_kuake_search]
    api_embed_total_params = [api_embedding_openai_api_key, api_embedding_openai_port,
                              api_embedding_azure_api_key, api_embedding_azure_endpoint, api_embedding_azure_engine]
    api_api_list.change(show_api_params_add_api, inputs=[api_api_list], outputs=[api_openai_params, api_azure_openai_params,
                        api_ernie_bot_params, api_ernie_bot_turbo_params, api_chatglm_params, api_spark_params, api_ali_params, api_selected_api])
    api_emb_refresh.click(refresh_embedding_directories,
                          outputs=[api_emb_model_list, api_doc1])
    use_knowledge.change(show_knowledge, inputs=[
                         use_knowledge], outputs=[api_local_knowledge])
    api_emb_model_list.change(emb_model_select, inputs=[
                              api_emb_api_list, api_emb_model_list], outputs=[api_emb_api_list, api_emb_model_list])
    api_emb_api_list.change(emb_api_select, inputs=[api_emb_api_list, api_emb_model_list], outputs=[
                            api_emb_openai_params, api_emb_azure_openai_params, api_emb_api_list, api_emb_model_list])
    api_emptymodelBtn.click(api_page_clear, outputs=[api_openai_chatbot, api_openai_knowledge_chatbot, api_azure_chatbot, api_azure_knowledge_chatbot, api_ernie_chatbot, api_ernie_knowledge_chatbot, api_ernie_turbo_chatbot, api_ernie_turbo_knowledge_chatbot, api_chatglm_chatbot,
                            api_chatglm_knowledge_chatbot, api_spark_chatbot, api_spark_knowledge_chatbot, api_ali_chatbot, api_ali_knowledge_chatbot, api_user_input, api_submitGroup, api_api_list, api_selected_api, use_knowledge, api_emb_model_list, api_emb_api_list, api_doc1])

    api_save.click(load_api_page_params, inputs=[api_api_list, api_emb_api_list, api_emb_model_list, api_doc1, api_k, api_score_threshold, api_chunk_size, api_chunk_conent, api_net, api_search, api_search_key, api_result_len, api_prompt, use_knowledge]+api_total_params+api_embed_total_params, outputs=[api_openai_chatbot, api_openai_knowledge_chatbot,
                   api_azure_chatbot, api_azure_knowledge_chatbot, api_ernie_chatbot, api_ernie_knowledge_chatbot, api_ernie_turbo_chatbot, api_ernie_turbo_knowledge_chatbot, api_chatglm_chatbot, api_chatglm_knowledge_chatbot, api_spark_chatbot, api_spark_knowledge_chatbot, api_ali_chatbot, api_ali_knowledge_chatbot, api_user_input, api_submitGroup], show_progress=True)
    api_emptyBtn.click(clear_history, outputs=[api_openai_chatbot, api_openai_knowledge_chatbot, api_azure_chatbot, api_azure_knowledge_chatbot, api_ernie_chatbot, api_ernie_knowledge_chatbot, api_ernie_turbo_chatbot,
                       api_ernie_turbo_knowledge_chatbot, api_chatglm_chatbot, api_chatglm_knowledge_chatbot, api_spark_chatbot, api_spark_knowledge_chatbot, api_ali_chatbot, api_ali_knowledge_chatbot], show_progress=True)

    api_submitBtn.click(parallel_api.call_openai, [user_input, api_openai_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_openai_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_openai_knowledge, [api_user_input, api_openai_knowledge_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_openai_knowledge_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_azure, [api_user_input, api_azure_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_azure_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_azure_knowledge, [api_user_input, api_azure_knowledge_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_azure_knowledge_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_ernie, [api_user_input, api_ernie_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_ernie_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_ernie_knowledge, [api_user_input, api_ernie_knowledge_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_ernie_knowledge_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_ernie_turbo, [api_user_input, api_ernie_turbo_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_ernie_turbo_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_ernie_turbo_knowledge, [api_user_input, api_ernie_turbo_knowledge_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_ernie_turbo_knowledge_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_chatglm, [api_user_input, api_chatglm_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_chatglm_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_chatglm_kmowledge, [api_user_input, api_chatglm_knowledge_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_chatglm_knowledge_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_spark, [api_user_input, api_spark_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_spark_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_spark_knowledge, [api_user_input, api_spark_knowledge_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_spark_knowledge_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_ali, [api_user_input, api_ali_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_ali_chatbot, api_user_input], show_progress=True)
    api_submitBtn.click(parallel_api.call_ali_knowledge, [api_user_input, api_ali_knowledge_chatbot,api_stream,net, search, search_key, result_len, prompt], [
        api_ali_knowledge_chatbot, api_user_input], show_progress=True)
