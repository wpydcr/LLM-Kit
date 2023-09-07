import gradio as gr
from utils.dl_data import *
import os
from utils.ui_utils import data_process
from utils.local_doc import local_doc_qa

real_path = os.path.split(os.path.realpath(__file__))[0]
new_path = os.path.join(real_path, "..", "models", "LLM")

def get_directories(path,unuse):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d not in unuse]

models = get_directories(new_path,['runs','vector_store'])
new_path = os.path.join(real_path, "..", "models", "Embedding")
embs = get_directories(new_path,[])

real_path = os.path.split(os.path.realpath(__file__))[0]
new_path = os.path.join(real_path, "..", "models", "Embedding")
embs = get_directories(new_path, [])
new_path = os.path.join(real_path, "..", "data", "documents")
docs = get_directories(new_path, [])

doc_qa=local_doc_qa()

data_pro=data_process()

embedding_api = ['openai','azure openai']

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

def data_page(localizer):
    with gr.Tab(localizer("LLM数据集制作")):
        with gr.Row():
            with gr.Column(scale=4):
                question1=gr.Textbox(lines=3, placeholder="Write Here...",label=localizer("问题："))
                answer1=gr.Textbox(lines=5, placeholder="Write Here...",label=localizer("答案："))
                with gr.Row():
                    save1 = gr.Button(localizer("保存"), variant="primary")
                    llm_sikp = gr.Button(localizer("跳过"))
                    gpt1 = gr.Button(localizer("单次gpt生成"))
                    start2 = gr.Button(localizer("批量gpt生成"))
                    stop2 = gr.Button(localizer("停止gpt生成"))
                    emptyBtn10 = gr.Button(localizer("清空json"))
                    emptyBtn11 = gr.Button(localizer("清空上传"))
                    back1 = gr.Button(localizer("撤回"))
                json_d1=gr.JSON(data_pro.json_dict,label=localizer("已保存数据"))
            with gr.Column(scale=1):
                    upload2 = gr.File(label=localizer("上传问题集(txt格式)"),file_count="single",file_types=['.txt'])
                    with gr.Accordion(localizer("gpt参数"),open=False):
                        openai_prompt=gr.Textbox(value=localizer("这是一个能够将文本翻译成中文的AI助手。请将引号中的文本翻译成简体中文。"),lines=3, placeholder="Write Here...",label=localizer("提示词"))
                        openai_api_key=gr.Textbox(lines=1, placeholder="Write Here...",label="openai_api_key:",type= 'password')
                        port1=gr.Textbox(lines=1, value='',label="VPN proxyPort:")
                        top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                        temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature:", interactive=True)
                        max_tokens = gr.Slider(0, 4096, value=0, step=64, label="Maximum length:", interactive=True)
                        timesleep = gr.Slider(0, 21, value=0.02, step=0.01, label=localizer("批量任务时间间隔(秒):"), interactive=True)
                    with gr.Accordion(localizer("下载数据"),open=False):
                        download_path1=gr.Textbox(lines=1, placeholder="Write Here...",label=localizer("保存文件名:"))
                        split_data11 = gr.Checkbox(label=localizer("是否划分验证集"))
                        download11 = gr.Button(localizer("下载数据"), variant="primary")
                        llm_upload_ok = gr.Textbox(lines=1, label=localizer("处理结果"))
    with gr.Tab(localizer("文档转LLM数据集")):
        with gr.Row():
            with gr.Column(scale=4):
                doc3=gr.Textbox(label=localizer("语料："))
                with gr.Row():
                    gpt3 = gr.Button(localizer("单次gpt生成"), variant="primary")
                    start3 = gr.Button(localizer("批量gpt生成"))
                    stop3 = gr.Button(localizer("停止gpt生成"))
                    emptyBtn30 = gr.Button(localizer("清空json"))
                    emptyBtn31 = gr.Button(localizer("清空上传"))
                    back3 = gr.Button(localizer("撤回"))
                json_d3=gr.JSON(data_pro.json_dict,label=localizer("已保存数据"))
                org_json_d3=gr.JSON(data_pro.json_dict,label=localizer("原始数据"))
            with gr.Column(scale=1):
                    upload3 = gr.File(label=localizer("上传文档(txt格式)"),file_count="single",file_types=['.txt'])
                    with gr.Accordion(localizer("gpt参数"),open=False):
                        openai_prompt3=gr.Textbox(value=localizer("这是一个能够根据文本内容，生成相关问答数据集的AI助手。请根据引号中的文本提炼出5个相关问题和回答。"),lines=3, placeholder="Write Here...",label=localizer("提示词"))
                        openai_api_key3=gr.Textbox(lines=1, placeholder="Write Here...",label="*openai_api_key:",type= 'password')
                        port3=gr.Textbox(lines=1, value='',label="*VPN proxyPort:")
                        top_p3 = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                        temperature3 = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature:", interactive=True)
                        max_tokens3 = gr.Slider(0, 4096, value=0, step=64, label="Maximum length:", interactive=True)
                        timesleep3 = gr.Slider(0, 21, value=0.02, step=0.01, label=localizer("批量任务时间间隔(秒):"), interactive=True)
                    with gr.Accordion(localizer("下载数据"),open=False):
                        download_path3=gr.Textbox(lines=1, placeholder="Write Here...",label=localizer("保存文件名:"))
                        split_data31 = gr.Checkbox(label=localizer("是否划分验证集"))
                        download33 = gr.Button(localizer("下载原始数据"))
                        download31 = gr.Button(localizer("下载数据"), variant="primary")
                        doc_to_llm_upload_ok = gr.Textbox(lines=1, label=localizer("处理结果"))
    with gr.Tab(localizer("LLM数据集转换")):
        question2=gr.Textbox(lines=1, label=localizer("问题key"),value='instruction')
        answer2=gr.Textbox(lines=1, label=localizer("回答key"),value='output')
        models2 = gr.Radio(['json(格式："[{},{}]")','jsonl(格式："{}{}")'], label=localizer("*数据集格式"),value='jsonl')
        models3 = gr.Radio(['新建','合并'], label=localizer("上传方式"),value='新建')
        upload1 = gr.File(label=localizer("上传数据集"),file_count="single",file_types=['.json','.jsonl','.data'])
        emptyBtn2 = gr.Button(localizer("清空上传"))
        out=gr.JSON(data_pro.upload_dict,label=localizer("预览后6行"))
        download_path2=gr.Textbox(lines=1, placeholder="Write Here...",label=localizer("保存文件名:"))
        split_data21 = gr.Checkbox(label=localizer("是否划分验证集"))
        download21 = gr.Button(localizer("下载数据"), variant="primary")
        llm_exchange_upload_ok = gr.Textbox(lines=1, label=localizer("处理结果"))
    with gr.Tab(localizer("Embedding数据集制作")):
        with gr.Row():
            with gr.Column(scale=4):
                sentence1=gr.Text(label=localizer("语料1:"))
                sentence2=gr.Text(label=localizer("语料2:"))
                with gr.Row():
                    embed_relate = gr.Button(value=localizer("相关"),variant="primary")
                    embed_unrelate = gr.Button(value=localizer("不相关"),)
                    embed_batch_generate = gr.Button(localizer("批量gpt生成"))
                    embed_stop_generate = gr.Button(localizer("停止gpt生成"))
                    embed_empty_json = gr.Button(localizer("清空json"))
                    embed_empty_upload = gr.Button(localizer("清空上传"))
                    embed_back = gr.Button(localizer("撤回"))
                embed_json_dict=gr.JSON(data_pro.embedding_json_dict,label=localizer("已保存数据"))
            with gr.Column(scale=1):
                 with gr.Accordion(label=localizer("上传数据集(txt格式)")):
                    embed_upload = gr.UploadButton(localizer("上传数据集"),file_types=['text'])
                 with gr.Accordion(localizer("gpt参数"),open=False):
                    openai_prompt_embed = gr.Textbox(value=localizer("这是一个能够判断两句话是否相关的AI助手。请判断引号中的两句话是否相关。如果相关输出1，不相关输出0，注意：只需要输出0或者1，不需要输出其他任何内容。"),lines=3, placeholder="Write Here...",label=localizer("提示词"))
                    openai_api_key_embed = gr.Textbox(lines=1, placeholder="Write Here...",label="*openai_api_key:",type= 'password')
                    port1_embed=gr.Textbox(lines=1, value='',label="*VPN Port:")
                    top_p_embed = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                    temperature_embed = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature:", interactive=True)
                    max_tokens_embed = gr.Slider(0, 2048, value=128, step=64, label="Maximum length:", interactive=True)
                    timesleep_embed = gr.Slider(0, 100, value=0.02, step=0.01, label=localizer("批量任务时间间隔(秒):"), interactive=True)
                 with gr.Accordion(localizer("下载数据"),open=False):
                    embed_download_path=gr.Textbox(lines=1, placeholder="Write Here...",label=localizer("数据文件夹名称："))
                    train_valid = gr.Checkbox(label=localizer("划分训练验证集"))
                    embed_download = gr.Button(localizer("下载数据"),variant='primary')
                    embed_upload_ok = gr.Textbox(lines=1, label=localizer("处理结果"))
    with gr.Tab(localizer("Embedding数据集转换")):
        embed_exchange_upload = gr.File(label=localizer("上传数据集"),file_count="single",file_types=['.data','.txt'])
        embed_exchange_emptyBtn = gr.Button(localizer("清空上传"))
        embed_exchange_models = gr.Radio(['新建','合并'], label=localizer("上传方式"),value='新建')
        embed_exchange_out=gr.JSON(data_pro.upload_dict,label=localizer("预览后6行"))
        embed_exchange_download_path=gr.Textbox(lines=1, placeholder="Write Here...",label=localizer("保存文件名:"))
        embed_exchange_train_valid = gr.Checkbox(label=localizer("划分训练验证集"))
        embed_exchange_download = gr.Button(localizer("下载数据"),variant='primary')
        embed_exchange_upload_ok = gr.Textbox(lines=1, label=localizer("处理结果"))
    with gr.Tab(localizer("知识库")):
        with gr.Tab("FAISS"):
            with gr.Tab(localizer("创建新的知识库")):
                vs_name = gr.Textbox(lines=1, label=localizer("库名"))
                with gr.Tab('Embedding API'):
                    emb_api_list = gr.Radio(embedding_api, show_label=False, value=None)
                    with gr.Accordion(localizer("openai参数"), open=True, visible=False) as openai_params:
                        openai_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*openai_api_key:", type='password')
                        openai_port = gr.Textbox(
                            lines=1, value='', label="*VPN proxyPort:")
                        openai_api_base = gr.Textbox(
                            lines=1, value='', label=localizer("API base:"))
                        openai_api_model = gr.Radio(
                                ['text-embedding-ada-002'], label=localizer("API模型"), value='text-embedding-ada-002')
                    with gr.Accordion(localizer("azure openai参数"), open=True, visible=False) as azure_openai_params:
                        azure_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*azure_api_key:", type='password')
                        azure_endpoint = gr.Textbox(
                            lines=1, value='', label="*endpoint:(azure openai)")
                        azure_engine = gr.Textbox(
                            lines=1, value='', label="*engine:(azure openai)")
                with gr.Tab(localizer("嵌入式模型")):
                    emb_model_list = gr.Radio(embs, show_label=False, value=None)
                emb_refresh = gr.Button(localizer("刷新"))
                files = gr.File(label=localizer("添加文件(支持md、pdf、txt、csv格式)"), file_count="multiple")
                upload = gr.Button(localizer("生成知识库"), variant="primary")
                upload_ok = gr.Textbox(lines=10, label=localizer("处理结果"))
            with gr.Tab(localizer("查询已有数据库内的文档")):

                doc0 = gr.Radio(docs, label=localizer("*本地的向量知识库"), value=None)
                with gr.Tab('Embedding API'):
                    create_emb_api_list = gr.Radio(embedding_api, show_label=False, value=None)
                    with gr.Accordion(localizer("openai参数"), open=True, visible=False) as create_openai_params:
                        create_openai_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*openai_api_key:", type='password')
                        create_openai_port = gr.Textbox(
                            lines=1, value='', label="*VPN proxyPort:")
                        create_openai_api_base = gr.Textbox(
                            lines=1, value='', label=localizer("API base:"))
                        create_openai_api_model = gr.Radio(
                                ['text-embedding-ada-002'], label=localizer("API模型"), value='text-embedding-ada-002')
                    with gr.Accordion(localizer("azure openai参数"), open=True, visible=False) as create_azure_openai_params:
                        create_azure_api_key = gr.Textbox(
                            lines=1, placeholder="Write Here...", label="*azure_api_key:", type='password')
                        create_azure_endpoint = gr.Textbox(
                            lines=1, value='', label="*endpoint:(azure openai)")
                        create_azure_engine = gr.Textbox(
                            lines=1, value='', label="*engine:(azure openai)")
                with gr.Tab(localizer("嵌入式模型")):
                    create_emb_model_list = gr.Radio(embs, show_label=False, value=None)
                refresh_edit = gr.Button(localizer("刷新"))
                search_button = gr.Button(localizer("查询"),variant="primary")
                file_in_doc = gr.CheckboxGroup([], label=localizer("库中的文档"), value=None)
                delete_selected_file = gr.Button(localizer("从库中删除选定的文档"))
                add_new_file = gr.File(label=localizer("添加文件(支持md、pdf、txt、csv格式)"), file_count="multiple")
                add_button = gr.Button(localizer("向库中添加文档"))

        with gr.Tab("MySQL"):
            # 数据库地址、端口、用户名、密码、数据库名称
            host = gr.Textbox(lines=1, label=localizer("*数据库地址"), value='localhost')
            user = gr.Textbox(lines=1, label=localizer("*用户名"), value='root')
            password = gr.Textbox(lines=1, label=localizer("*密码"),type= 'password', value='123456')
            port = gr.Textbox(lines=1, label=localizer("*端口"), value='3306')
            connect = gr.Button(localizer("连接数据库"), variant="primary")
            database = gr.Radio(label=localizer("选择数据库"),value=None)
            new_database = gr.Textbox(lines=1, label=localizer("新建数据库"), placeholder="Write Here...", visible=False)
            # 上传文件，支持多文件上传，文件类型为csv、txt、excel
            mysql_files = gr.File(label=localizer("添加文件(支持txt、csv、excel格式)"),file_count="multiple")
            # 上传按钮
            mysql_upload = gr.Button(localizer("生成知识库"))
            # 上传结果
            mysql_upload_ok=gr.Textbox(lines=10, label=localizer("处理结果"))
        # with gr.Tab("other"):
        #     pass
    
    create_emb_total_params = [create_openai_api_key,create_openai_port,create_openai_api_base, create_openai_api_model, create_azure_api_key,create_azure_endpoint,create_azure_engine]
    search_button.click(doc_qa.handle_database_selected, inputs=[create_emb_api_list,create_emb_model_list,doc0]+create_emb_total_params,outputs= file_in_doc)
    delete_selected_file.click(doc_qa.handle_vector_database_file_delete, inputs=[file_in_doc,doc0],outputs=file_in_doc)
    add_button.click(doc_qa.handle_add_file_to_vector_database, inputs=[add_new_file,doc0],outputs=file_in_doc)
    emb_refresh.click(doc_qa.refresh_emb,  outputs=[emb_model_list])
    refresh_edit.click(doc_qa.refresh_vector,outputs=[doc0,create_emb_model_list])

    save1.click(data_pro.save_data, [question1, answer1],  outputs=[json_d1,question1,answer1],show_progress=True)
    llm_sikp.click(data_pro.skip_qa,[answer1],outputs=[question1,answer1])
    gpt1.click(data_pro.ones_openai, [question1,openai_api_key,temperature,max_tokens,top_p,openai_prompt,port1],  outputs=[answer1],show_progress=True)
    upload2.upload(data_pro.upload_data, [upload2],  outputs=[question1],show_progress=True)
    download11.click(data_pro.dl_jsonl1, [download_path1,split_data11],[llm_upload_ok],show_progress=True)
    emptyBtn10.click(data_pro.reset_state, [],[json_d1])
    emptyBtn11.click(data_pro.reset_upload, [],[question1])
    back1.click(data_pro.back_state, [],[json_d1])
    start2.click(data_pro.start_openai, [openai_api_key,temperature,max_tokens,top_p,openai_prompt,port1,timesleep],[json_d1])
    stop2.click(data_pro.stop_openai, [],[json_d1,question1])

    gpt3.click(data_pro.ones_openai_doc, [doc3,openai_api_key3,temperature3,max_tokens3,top_p3,openai_prompt3,port3],  outputs=[json_d3, org_json_d3],show_progress=True)
    upload3.upload(data_pro.upload_data, [upload3],  outputs=[doc3],show_progress=True)
    download31.click(data_pro.dl_jsonl1, [download_path3,split_data31],[doc_to_llm_upload_ok],show_progress=True)
    download33.click(data_pro.dl_jsonl1,[download_path3,split_data31],[doc_to_llm_upload_ok],show_progress=True)
    emptyBtn30.click(data_pro.reset_state_doc, [],[json_d3,org_json_d3])
    emptyBtn31.click(data_pro.reset_upload, [],[doc3])
    back3.click(data_pro.back_state, [],[json_d3])
    start3.click(data_pro.start_openai_doc, [openai_api_key3,temperature3,max_tokens3,top_p3,openai_prompt3,port1,timesleep3],[json_d3,org_json_d3])
    stop3.click(data_pro.stop_openai_doc, [],[json_d3,org_json_d3,doc3])
    
    total_faiss_params = [files,vs_name,openai_api_key,openai_port, openai_api_base, openai_api_model, azure_api_key,azure_endpoint,azure_engine]
    upload1.upload(data_pro.upload_data_out, [upload1,question2,answer2,models2,models3],  outputs=[out],show_progress=True)
    emptyBtn2.click(data_pro.reset_upload_out, [],[out])
    download21.click(data_pro.dl_jsonl2, [download_path2,split_data21],[llm_exchange_upload_ok],show_progress=True)
    upload.click(doc_qa.upload_data,[emb_api_list,emb_model_list]+total_faiss_params,[upload_ok],show_progress=True)


    embed_relate.click(data_pro.save_embed_data,[sentence1,sentence2,embed_relate],outputs=[embed_json_dict,sentence1,sentence2],show_progress=True)
    embed_unrelate.click(data_pro.save_embed_data,[sentence1,sentence2,embed_unrelate],outputs=[embed_json_dict,sentence1,sentence2],show_progress=True)
    embed_back.click(data_pro.back_embed_json,[],outputs=[embed_json_dict],show_progress=True)
    embed_download.click(data_pro.dl_embed,[embed_download_path,train_valid],[embed_upload_ok],show_progress=True)
    embed_empty_json.click(data_pro.empty_embed_json,[],outputs=[embed_json_dict],show_progress=True)
    embed_empty_upload.click(data_pro.empty_embed_upload,[],outputs=[sentence1,sentence2],show_progress=True)
    embed_upload.upload(data_pro.upload_embed_data,[embed_upload],[sentence1,sentence2],show_progress=True)
    embed_batch_generate.click(data_pro.start_openai_embed,[openai_api_key_embed,temperature_embed,max_tokens_embed,top_p_embed,openai_prompt_embed,port1_embed,timesleep_embed],[embed_json_dict,sentence1,sentence2],show_progress=True)
    embed_stop_generate.click(data_pro.stop_openai_embed,[],[],show_progress=True)

    embed_exchange_upload.upload(data_pro.upload_embed_exchange_data,[embed_exchange_upload,embed_exchange_models],[embed_exchange_out],show_progress=True)
    embed_exchange_emptyBtn.click(data_pro.empty_embed_exchange_upload,[],[embed_exchange_out],show_progress=True)
    embed_exchange_download.click(data_pro.dl_embed,[embed_exchange_download_path,embed_exchange_train_valid],outputs=[embed_exchange_upload_ok],show_progress=True)

    connect.click(data_pro.connect_mysql,[host,user,password,port],[database],show_progress=True)
    database.change(data_pro.change_database,[database],[new_database],show_progress=True)
    mysql_upload.click(data_pro.mysql_upload,[host,user,password,port,database,new_database,mysql_files],[mysql_upload_ok],show_progress=True)

    emb_model_list.change(emb_model_select, inputs=[emb_api_list,emb_model_list], outputs=[emb_api_list,emb_model_list])
    emb_api_list.change(emb_api_select, inputs=[emb_api_list,emb_model_list], outputs=[openai_params,azure_openai_params,emb_api_list,emb_model_list])

    create_emb_model_list.change(emb_model_select, inputs=[create_emb_api_list,create_emb_model_list], outputs=[create_emb_api_list,create_emb_model_list])
    create_emb_api_list.change(emb_api_select, inputs=[create_emb_api_list,create_emb_model_list], outputs=[create_openai_params,create_azure_openai_params,create_emb_api_list,create_emb_model_list])

