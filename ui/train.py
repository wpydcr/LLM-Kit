import gradio as gr
import os
from utils.ui_utils import llm_train
from utils.ui_utils import embedding_train_utils
import json
from utils.embedding_val import embedding_visualization_plot


real_path = os.path.split(os.path.realpath(__file__))[0]
def get_file(path):
    return [d for d in os.listdir(path) if d.endswith('.json') or d.endswith('.jsonl')]

def get_directories(path,unuse):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d not in unuse]

embedding_api = ['openai','azure openai']

new_path = os.path.join(real_path, "..", "models", "LLM")
models = get_directories(new_path,['runs','vector_store'])

new_path = os.path.join(real_path, "..", "models", "Embedding")
embs = get_directories(new_path,[])

new_path = os.path.join(real_path, "..", "data",'modeldata', "LLM")
llm_datas = get_directories(new_path,[])

new_path = os.path.join(real_path, "..", "data",'modeldata', "Embedding")
emb_datas = get_directories(new_path,[])

model_arch = json.load(open(os.path.join(real_path, "..","data","config", "embedding_train", "model_arch.json"), 'r', encoding='utf-8'))
model_arch_list = list(set([model_arch[i]['model_arch'] for i in model_arch.keys()]))

def switch_model_arch(model):
    if model_arch.get(model,None) is None:
        return gr.update(value=None,interactive=True)
    return gr.update(value=model_arch[model]['model_arch'],interactive=False)

def add_models(api_list,model_list):
    return gr.update(value=' | '.join((api_list+model_list)))

def show_emb_params_add_api(api_list,model_list):
    res = []
    if 'openai' in api_list:
        res.append(gr.update(visible=True))
    else:
        res.append(gr.update(visible=False))
    if 'azure openai' in api_list:
        res.append(gr.update(visible=True))
    else:
        res.append(gr.update(visible=False))
    return res[0],res[1],gr.update(value=' | '.join((api_list+model_list)))

def load_emb_val_params(api_list,model_list,*args):
    from webui import  localizer
    if api_list is None and model_list is None:
        raise localizer("未选择API或模型")
    params = {}
    if 'openai' in api_list:
        params['openai'] = {
            'api_key':args[0],
            'port':args[1]
        }
    if 'azure openai' in api_list:
        params['azure openai'] = {
            'api_key':args[2],
            'endpoint':args[3],
            'engine':args[4]
        }
    
    return embedding_vis.plot(api_list,model_list,params)

embedding_vis = embedding_visualization_plot()
embedding_train = embedding_train_utils()
llm_train_cofig = llm_train()

def train_page(localizer):
    with gr.Tab("LLM"):
            with gr.Row():
                with gr.Column(scale=1):

                    LLM_models1 = gr.Radio(models, label=localizer("*选择模型"),value=None)
                    LLM_data1 = gr.Radio(llm_datas, label=localizer("*选择数据"),value=None)
                    lora = gr.Radio(['Lora',localizer("全量")], label=localizer("*微调方式"),value=None)
                    lora_checkpoint = gr.Radio(label='Checkpoint', value=None, interactive=True)
                    lora_rank = gr.Slider(1,20,value=8,step=1,label='lora_rank')
                    lora_use_8bit_4bit = gr.Radio(['8 bit', '4 bit', None], label="lora use 8/4 bit", value=None)

                    with gr.Accordion('deepspeed', open=False):
                        deepspeed = gr.Radio([True,False],label=localizer("使用deepspeed(仅支持Linux)"),value=False)
                        if llm_train.detect_OS():
                            deepspeed.interactive = False
                        with gr.Accordion('deepspeed setting', open=False):
                            LLM_mixed_precision = gr.Radio(['no', 'fp16', 'bf16'], label="Choose mixed precision", value="no",
                                                       interactive=False)

                            with gr.Accordion('mixed precision setting', open=False):
                                LLM_enabled = gr.Radio([True,False], label="enable", value=True)
                                LLM_loss_scale= gr.Slider(0, 5, value=0, step=0.1, label="loss scale")
                                LLM_loss_window= gr.Slider(100, 5000, value=1000, step=100, label="loss scale window")
                                LLM_initial_scale_power= gr.Slider(0, 50, value=2, step=1, label="initial scale power")
                                LLM_hysteresis= gr.Slider(0, 10, value=2, step=1, label="hysteresis")
                                LLM_min_loss_scale= gr.Slider(0, 10, value=2, step=1, label="loss scale")

                                with gr.Accordion('optimizer', open=False):
                                    LLM_optimizer_type = gr.Radio(["AdamW"], label="type", value="AdamW",interactive=True)
                                    with gr.Accordion('params', open=True):
                                        LLM_optimizer_params_lr= gr.Radio(["auto"], label="lr",value="auto",interactive=True)
                                        LLM_optimizer_params_weight_decay= gr.Radio(["auto"], label="weight_decay",value="auto",interactive=True)

                                with gr.Accordion('scheduler', open=False):
                                    LLM_scheduler_type = gr.Radio(["WarmupLR", "inverse_sqrt"], label="type", value="WarmupLR", interactive=True)
                                    with gr.Accordion('params', open=True):
                                        LLM_scheduler_warmup_min_lr = gr.Radio(["auto"], label="warmup min lr", value="auto",interactive=True)
                                        LLM_scheduler_warmup_max_lr = gr.Radio(["auto"], label="warmup max lr", value="auto",interactive=True)
                                        LLM_scheduler_warmup_num_steps = gr.Radio(["auto"], label="warmup num steps", value="auto",interactive=True)

                                with gr.Accordion('zero_optimization', open=False):
                                    LLM_zero_optimization_stage = gr.Slider(1, 10, value=3, step=1, label="stage", interactive=True)
                                    with gr.Accordion('offload_optimizer', open=True):
                                        LLM_zero_optimization_offload_optimizer_device = gr.Radio(["cpu", "gpu"], label="device", value="cpu",interactive=True)
                                        LLM_zero_optimization_pin_offload_optimizer_memory = gr.Radio([True,False], label="pin memory", value=True, interactive=True)
                                    with gr.Accordion('offload_param', open=True):
                                        LLM_zero_optimization_offload_param_device = gr.Radio(["cpu", "gpu"], label="device", value="cpu",interactive=True)
                                        LLM_zero_optimization_pin_offload_param_memory = gr.Radio([True,False], label="pin memory", value=True,interactive=True)

                            LLM_zero_optimization_overlap_comm = gr.Radio([True,False], label="overlap comm", value=True,interactive=True)
                            LLM_zero_optimization_contiguous_gradients = gr.Radio([True,False], label="contiguous gradients", value=True,interactive=True)
                            LLM_zero_optimization_overlap_reduce_bucket_size = gr.Radio(["auto"], label="reduce bucket ize", value="auto",interactive=True)
                            LLM_zero_optimization_stage3_prefetch_bucket_size = gr.Radio(["auto"], label="stage3 prefetch bucket_size",value="auto",interactive=True)
                            LLM_zero_optimization_stage3_stage3_param_persistence_threshold = gr.Radio(["auto", "other"], label="stage3 param persistence threshold",value="auto",interactive="auto")
                            LLM_zero_optimization_sub_group_size = gr.Slider(100000000, 10000000000, value=1000000000, step=10000000, label="sub group size", interactive=True)
                            LLM_zero_optimization_stage3_max_live_parameters = gr.Slider(100000000, 10000000000, value=1000000000, step=10000000, label="stage3 max live parameters", interactive=True)
                            LLM_zero_optimization_stage3_max_reuse_distance = gr.Slider(100000000, 10000000000, value=1000000000, step=10000000, label="stage3 max reuse distance", interactive=True)
                            LLM_zero_optimization_stage3_gather_16bit_weights_on_model_save = gr.Radio([True,False], label="stage3 gather 16bit weights on model save",value=True,interactive=True)

                        LLM_gradient_clipping = gr.Radio(["auto"], label="gradient clipping",value="auto", interactive=True)
                        LLM_steps_per_print = gr.Radio(["auto"], label="steps per print",value="auto", interactive=True)
                        LLM_train_batch_size = gr.Radio(["auto"], label="train batch size (Assume same batch size for eval, could be modified in train.py)",value="auto", interactive=True)
                        LLM_train_micro_batch_size_per_gpu = gr.Radio(["auto"], label="train micro batch size per gpu",value="auto", interactive=True)
                        LLM_wall_clock_breakdown = gr.Radio([True,False], label="wall clock breakdown",value=False, interactive=True)
                with gr.Column(scale=2):
                    with gr.Accordion(localizer("多设备设置"), open=False):
                        LLM_compute_environment = gr.Radio(["LOCAL_MACHINE", "REMOTE_MACHINE"], label="Compute Environment", value="LOCAL_MACHINE")
                        LLM_machine_rank = gr.Slider(0, 15, value=0, step=1, label="machine rank", interactive=0)
                        LLM_num_machines= gr.Slider(0, 15, value=1, step=1, label="machine number")
                        LLM_rdzv_backend= gr.Radio(["static","dynamic", "Centralized", "decentralized"], label="rendezvous backend",
                                                   value="static", interactive=False)
                        LLM_same_network= gr.Radio([True,False], label="same network",value=True)

                    with gr.Accordion(localizer("设备设置"), open=True):
                        LLM_device = gr.Radio(['cpu', 'gpu'], label="Choose device", value='cpu')
                        _, GPU_count = llm_train.get_avaliable_gpus()
                        LLM_devices = gr.CheckboxGroup(GPU_count, label="Choose GPUs", value=None, interactive=False)

                        with gr.Accordion('TPU', open=False):
                            LLM_tpu_env= gr.Radio(["[]"], label="tpu environment",value=[])
                            LLM_other_env_box = gr.Textbox(label="Enter the environment, press enter to submit:", visible=False)
                            LLM_tpu_use_cluster= gr.Radio([True,False], label="use cluster",value=False)
                            LLM_tpu_use_sudo= gr.Radio([True,False], label="use sudo",value=False)
                with gr.Column(scale=3):
                    with gr.Accordion('Train Settings', open=True):
                        LLM_downcast_bf16 = gr.Radio(["no", "yes"], label="downcast bf16", value="no")
                        LLM_main_training_function = gr.Radio(["main"], label="main training function",value="main")
                        LLM_other_function_box = gr.Textbox(label="Enter the function name, press enter to submit:",visible=False)
                        LLM_batch_size = gr.Slider(1, 1024, value=1, step=1, label="per_device_train_batch_size", interactive=True)
                        LLM_max_steps = gr.Slider(10, 104000, value=100, step=10, label="max_steps", interactive=True)
                        LLM_save_steps = gr.Slider(1, 5000, value=10, step=10, label="save_steps", interactive=True)
                        LLM_learning_rate = gr.Slider(1e-6, 1e-3, value=1e-5, step=None, label="learning_rate", interactive=True)
                        LLM_logging_steps = gr.Slider(10, 5000, value=10, step=10, label="logging_steps", interactive=True)
                        LLM_epochs = gr.Slider(1, 1000, value=2, step=1, label="train epochs", interactive=True)
                        LLM_weight_decay = gr.Slider(0, 1, value=0, step=0.001, label="wight_decay", interactive=True)
                        LLM_gradient_accumulation_steps = gr.Slider(1, 20, value=5, step=1, label="gradient accumulation steps", interactive=True)
            with gr.Row():
                refresh_LLM = gr.Button(localizer("刷新"))
                # with gr.Column(scale=1):
                LLM_start = gr.Button(localizer("开始训练"), variant="primary")
                # with gr.Column(scale=2):
                LLM_end = gr.Button(localizer("停止训练"), variant="secondary")
            with gr.Row():
                with gr.Column():
                    Lora_out = gr.Textbox(placeholder="epoch:0.0  loss：0.0 ", lines=10, label="training")
    with gr.Tab(localizer("嵌入式模型")):
        with gr.Tab(localizer("嵌入式模型训练")):
            with gr.Row():
                with gr.Column(scale=1):
                    embed_emb0 = gr.Radio(embs, label="*Choice Embedding",value=None)
                    embed_arch = gr.Radio(model_arch_list, label="*Choice model architecture",value=None)
                with gr.Column(scale=1):
                    embed_data1 = gr.Radio(emb_datas, label=localizer('*选择数据'),value=None)
                    embed_device1 = gr.Radio(['cpu','single_gpu','multi_gpu'], label="Choice device",value='single_gpu')
                    embed_save_dir=gr.Textbox(lines=1, placeholder="Write Here...",label=localizer("模型保存文件夹(仅支持英文路径)"))
                with gr.Column(scale=2):
                    with gr.Accordion(localizer("参数"),open=False):
                        with gr.Row():
                            with gr.Column(scale=1):
                                embed_batch_size = gr.Slider(1, 1024, value=4, step=1, label="per_device_train_batch_size", interactive=True)
                                embed_epochs = gr.Slider(1, 1024, value=10, step=1, label="epochs", interactive=True)
                                embed_weight_decay = gr.Slider(0,1, value=0.01, step=0.01, label="weight_decay", interactive=True)
                                embed_warmup_ratio = gr.Slider(0,1, value=0.1, step=0.01, label="warmup_ratio", interactive=True)
                                embed_eps = gr.Slider(0,1, value=1e-6, step=1e-6, label="eps", interactive=True)
                            with gr.Column(scale=1):
                                embed_gradient_accumulation_steps = gr.Slider(1,100, value=1, step=1, label="gradient_accumulation_steps", interactive=True)
                                embed_max_steps = gr.Slider(10, 104000, value=100, step=10, label="max_steps", interactive=True)
                                embed_save_steps = gr.Slider(10, 5000, value=10, step=10, label="save_steps", interactive=True)
                                embed_learning_rate = gr.Slider(1e-8, 1e-3, value=1e-4, step=None, label="learning_rate", interactive=True)
                                embed_logging_epochs = gr.Slider(1, 5000, value=10, step=10, label="logging_epochs", interactive=True)
            with gr.Row():
                refresh_data = gr.Button(localizer("刷新"))
                embed_start = gr.Button(localizer("开始训练"), variant="primary")
                embed_stop = gr.Button(localizer("停止训练"), variant="secondary")
            with gr.Row():
                embed_out = gr.Textbox(placeholder="epoch：0.0  loss：0.0 ", lines=10, label="training")

        with gr.Tab(localizer("嵌入式模型验证")):
            with gr.Tab(localizer("API列表")):
                api_list = gr.CheckboxGroup(embedding_api, show_label=False, value=None, interactive=True)
                with gr.Accordion(localizer("openai参数"), open=False, visible=False) as emb_val_openai_params:
                    openai_api_key = gr.Textbox(
                        lines=1, placeholder="Write Here...", label="*openai_api_key:", type='password')
                    openai_port = gr.Textbox(
                        lines=1, value='', label="*VPN proxyPort:")
                with gr.Accordion(localizer("azure openai参数"), open=False, visible=False) as emb_val_azure_openai_params:
                    azure_api_key = gr.Textbox(
                        lines=1, placeholder="Write Here...", label="*azure_api_key:", type='password')
                    azure_endpoint = gr.Textbox(
                        lines=1, value='', label="*endpoint:(azure openai)")
                    azure_engine = gr.Textbox(
                        lines=1, value='', label="*engine:(azure openai)")
            with gr.Tab(localizer("模型列表")):
                model_list = gr.CheckboxGroup(embs, show_label=False, value=None, interactive=True)
            selected_models = gr.Textbox(label=localizer("已选择的模型"), lines=1, value=None, interactive=False)
            Refresh = gr.Button(localizer("刷新本地嵌入模型库"))
            upload4 = gr.UploadButton(localizer("上传数据文件(支持[json,csv,docx,txt,pdf]格式)"))
            text = gr.Textbox(label=localizer("文件上传确认"))
            dispalyBtn = gr.Button(localizer("绘制可视化图片"), variant="primary")
            show =gr.Plot()

    refresh_LLM.click(llm_train_cofig.handle_refresh_LLM,inputs = [], outputs=[LLM_models1,LLM_data1])
    refresh_data.click(llm_train_cofig.handle_refresh_embd_and_data,outputs=[embed_emb0,embed_data1])
    LLM_device.change(llm_train_cofig.switch_CPU_GPU,inputs= LLM_device,outputs=LLM_devices)
    deepspeed.change(llm_train_cofig.switch_deepspeed,inputs= deepspeed, outputs=LLM_mixed_precision)
    lora.change(llm_train_cofig.switch_checkpoint, inputs=[lora,LLM_models1], outputs=[lora_checkpoint])
    LLM_models1.change(llm_train_cofig.switch_checkpoint, inputs=[lora,LLM_models1], outputs=[lora_checkpoint])
    LLM_start.click(llm_train_cofig.post_train_request,
                    inputs=[deepspeed, LLM_mixed_precision, LLM_compute_environment, LLM_machine_rank,
                           LLM_num_machines, LLM_rdzv_backend, LLM_same_network, LLM_device, LLM_devices, LLM_tpu_env,
                           LLM_tpu_use_cluster, LLM_tpu_use_sudo, LLM_downcast_bf16, LLM_main_training_function,
                           LLM_enabled, LLM_loss_scale, LLM_loss_window, LLM_initial_scale_power, LLM_hysteresis,
                           LLM_min_loss_scale, LLM_optimizer_type, LLM_optimizer_params_lr,
                           LLM_optimizer_params_weight_decay, LLM_scheduler_type, LLM_scheduler_warmup_min_lr,
                           LLM_scheduler_warmup_max_lr, LLM_scheduler_warmup_num_steps, LLM_zero_optimization_stage,
                           LLM_zero_optimization_offload_optimizer_device,
                           LLM_zero_optimization_pin_offload_optimizer_memory,
                           LLM_zero_optimization_offload_param_device, LLM_zero_optimization_pin_offload_param_memory,
                           LLM_zero_optimization_overlap_comm, LLM_zero_optimization_contiguous_gradients,
                           LLM_zero_optimization_overlap_reduce_bucket_size,
                           LLM_zero_optimization_stage3_prefetch_bucket_size,
                           LLM_zero_optimization_stage3_stage3_param_persistence_threshold,
                           LLM_zero_optimization_sub_group_size, LLM_zero_optimization_stage3_max_live_parameters,
                           LLM_zero_optimization_stage3_max_reuse_distance,
                           LLM_zero_optimization_stage3_gather_16bit_weights_on_model_save,
                           LLM_gradient_accumulation_steps, LLM_gradient_clipping, LLM_steps_per_print,
                           LLM_train_batch_size, LLM_train_micro_batch_size_per_gpu, LLM_wall_clock_breakdown,
                           LLM_models1, LLM_data1, lora,lora_checkpoint,lora_rank, lora_use_8bit_4bit, LLM_batch_size,
                           LLM_learning_rate, LLM_save_steps, LLM_max_steps, LLM_weight_decay, LLM_epochs,
                           LLM_logging_steps],
                            outputs=Lora_out)
    LLM_end.click(llm_train_cofig.post_stop_request,inputs=[],outputs=[])
                    
    
    embed_start.click(embedding_train.start_train,[embed_emb0,embed_arch,embed_data1,embed_device1,embed_batch_size,embed_max_steps,embed_save_steps,embed_learning_rate,embed_logging_epochs,embed_save_dir,embed_epochs,embed_weight_decay,embed_warmup_ratio,embed_eps,embed_gradient_accumulation_steps],[embed_out],show_progress=True)
    embed_stop.click(embedding_train.stop_train,[],[],show_progress=True)

    total_params = [openai_api_key,openai_port,azure_api_key,azure_endpoint,azure_engine]

    Refresh.click(embedding_vis.refresh_directories, outputs=[model_list])
    upload4.upload(embedding_vis.upload_data, inputs=[upload4],outputs=text,show_progress=True)
    dispalyBtn.click(load_emb_val_params,inputs=[api_list,model_list]+total_params,outputs=show)

    embed_emb0.change(switch_model_arch,inputs=[embed_emb0],outputs=[embed_arch])


    api_list.change(show_emb_params_add_api,inputs=[api_list,model_list],outputs=[emb_val_openai_params,emb_val_azure_openai_params,selected_models])

    model_list.change(add_models,inputs=[api_list,model_list],outputs=[selected_models])