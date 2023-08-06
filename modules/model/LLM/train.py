import sys
import os
sys.path.append(os.path.realpath(__file__).split('caio_web')[0]+'caio_web')
print(sys.path)
import numpy as np
import argparse
import logging
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
real_path = os.path.split(os.path.realpath(__file__))[0]

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    SchedulerType,
    get_scheduler
)
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
# from accelerate.utils import DummyOptim, DummyScheduler, set_seed  #DummyOptim and DummyScheduler cannot import on Windows, to be noticed
from accelerate.utils import set_seed
# sys.path.append(os.path.join(real_path, '..', '..'))
from utils.utils import get_model_tokenizer, get_preprocess_datacollator, get_lora_model, copy_custom_files
import json
# from peft import prepare_model_for_int8_training
# 需要 pip install git+https://github.com/huggingface/peft.git
from peft import prepare_model_for_kbit_training

logger = get_logger(__name__)
def evaluate(args, model, eval_dataloader, accelerator):
    """
    Evaluate the model on the evaluation dataset.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): The model to evaluate.
        eval_dataloader (torch.utils.data.DataLoader): The evaluation dataloader.
        accelerator (accelerate.Accelerator): The accelerator object.

    Returns:
        tuple: Tuple containing the perplexity and evaluation loss.
    """
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(
            loss.repeat(args.per_device_eval_batch_size)))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss

def get_train_valid_data(dir):
    """
    Get the path of the training and validation data files.

    Args:
        dir (str): The directory path.

    Returns:
        tuple: Tuple containing the names of the training and validation data files.
    """
    files = os.listdir(os.path.join('data','modeldata','LLM',dir))
    train_data = ""
    valid_data = ""
    for file in files:
        if 'train' in file:
            train_data = file
        elif 'valid' in file:
            valid_data = file
    return train_data, valid_data


def read_json(path):
    with open(path) as file:
        json_data = json.load(file)
    return json_data

def parse_args():


    path = os.path.join(real_path, "..","..", "..","data", "config", "train_config", "model_hyperparam_config.json")

    # path = "model_hyperparam_config.json"
    args_to_load = read_json(path)

    train_data, valid_data = get_train_valid_data(args_to_load['data'])


    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--train_file_name", type=str, default=train_data, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file_name", type=str, default=valid_data, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=args_to_load["model name"],
        choices=["chatglm-6b", "moss-moon-003-sft", "phoenix-inst-chat-7b", "Guanaco", "baichuan-vicuna-chinese-7b", "chatglm2-6b", "chatglm2-6b-32k", "Baichuan-13B-Chat", "internlm-chat-7b-8k", "chinese-alpaca-2-7b"], 
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="User defined prefix name."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=args_to_load["training batch size(per device)"],
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=args_to_load["training batch size(per device)"],
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=args_to_load["learning rate"],
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=args_to_load["weight decay"], help="Weight decay to use.")

    parser.add_argument("--num_train_epochs", type=int, default=args_to_load["train epochs"],
                        help="Total number of training epochs to perform.")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=args_to_load["gradient accumulation steps"],
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default=args_to_load["lr scheduler type"],
        help="The scheduler type to use.",
        choices=["constant_with_warmup", "inverse_sqrt"], #WarmupLR in deepspeed config?
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="A seed for reproducible training.")

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=0,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument(
        "--steps_per_print",
        type=int,
        default=args_to_load["logging steps"],
        help="Printing loss at the end of every n steps.",
    )
 
    parser.add_argument(
        "--use_lora",
        default=True if args_to_load["use lora"]=='Lora' else False,
        action="store_true",
        help="Whether to use lora.",
    )

    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=args_to_load['lora checkpoint'],
        help="Whether to load lora checkpoint.",
    )

    parser.add_argument(
        "--use_8bit",
        default=True if args_to_load["use lora 8bit 4bit"] == '8 bit' else False,
        action="store_true",
        help="Whether to use 8bit trainnig. Only for lora finetunnig.",
    )

    parser.add_argument(
        "--use_4bit",
        default=True if args_to_load["use lora 8bit 4bit"] == '4 bit' else False,
        action="store_true",
        help="Whether to use 4bit trainnig. Only for lora finetunnig.",
    )

    parser.add_argument(
        "--gradient_checkpoint",
        action="store_true",
        help="Whether to use gradient checkpoint."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=args_to_load['lora rank'],
        help="Lora rank.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=args_to_load["save steps"],
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=args_to_load["max steps"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=args_to_load["output dir"],
    )
    args = parser.parse_args()
    args.model_path = os.path.join(real_path, '..',"..", "..", "models", "LLM", args.model_name)
    

    # Sanity checks
    args.train_file = os.path.join(real_path,'..', "..", "..", "data", "modeldata", "LLM", args_to_load['data'],args.train_file_name)
    if args.validation_file_name is not None:
        args.validation_file = os.path.join(real_path, '..',"..", "..", "data", "modeldata", "LLM",args_to_load['data'],
                                            args.validation_file_name)
    
    return args

# New Code #

def main():
    """
       Main function to run the training process, multiple models adapted, including chatlgm, moss, phoenix...
       To train with accelerate config, see train_luancher.py
       """
    # manual_launch_main()

    args = parse_args()

    accelerator = Accelerator(dispatch_batches=False)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.ERROR,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    
    accelerator.wait_for_everyone()

    # print("before get_model_tokenizer", torch.cuda.memory_allocated()/1024**3/1024**3)  # current allocated memory in MB
    # print(torch.cuda.memory_reserved()/1024**3)  # current reserved memory in MB

    # model, tokenizer = get_model_tokenizer(args.model_path, use_8bit=args.use_8bit, max_length=args.max_length, use_deepspeed=accelerator.state.deepspeed_plugin is not None) Should this be loaded from config?
    if not os.path.exists(os.path.join(real_path,'..','..','..','flag.txt')):
        model, tokenizer = get_model_tokenizer(args.model_path, use_8bit=args.use_8bit, use_4bit=args.use_4bit, max_length=args.max_length, use_deepspeed=accelerator.state.deepspeed_plugin is not None)
    else:
        print('load model stop')
        return

    model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpoint:
        model.gradient_checkpointing_enable()
    if args.use_lora:
        if args.use_8bit or args.use_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpoint)
        else:
            model.enable_input_require_grads()
        model = get_lora_model(model, args.model_path, args.lora_rank,args.lora_checkpoint)

    preprocess, data_collator = get_preprocess_datacollator(
        args.model_path)

    if not os.path.isdir(args.train_file):
        data_files = {}
        dataset_args = {}
        data_files["train"] = args.train_file
        if args.validation_file_name is not None:
            data_files["validation"] = args.validation_file
        dataset_args["streaming"] = True
        raw_datasets = load_dataset(
            "json", data_files=data_files, **dataset_args)

        # 统一命名,目前仅支持单轮对话数据集,以及Belle group提供的多轮对话数据
        raw_datasets = raw_datasets.rename_column("instruction", "question")
        raw_datasets = raw_datasets.rename_column("output", "answer")
        with accelerator.main_process_first():
            lm_datasets = raw_datasets.map(
                lambda examples: preprocess(examples, tokenizer, args.max_length),
            )

        train_dataset = lm_datasets["train"]
        if args.validation_file_name is not None:
            eval_dataset = lm_datasets["validation"]
    else:
        train_dataset = load_from_disk(args.train_file)
        if args.validation_file_name is not None:
            eval_dataset = load_from_disk(args.validation_file)

    train_dataset = train_dataset.shuffle(seed=args.seed)

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=lambda features: data_collator(features, tokenizer),
        batch_size=args.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers
    )
    if args.validation_file_name is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=lambda features: data_collator(features, tokenizer),
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.preprocessing_num_workers
        )


        eval_dataloader = accelerator.prepare(eval_dataloader)
    else:
        eval_dataloader = None

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    if len(optimizer_grouped_parameters[1]["params"]) == 0:
        optimizer_grouped_parameters = optimizer_grouped_parameters[:1]
        
    # New Code #
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        #else DummyOptim
        else False
    )
    optimizer = optimizer_cls(
        optimizer_grouped_parameters, lr=args.learning_rate)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Scheduler and math around the number of training steps.

    # New Code
    # Get gradient accumulation steps from deepspeed config if available
    if accelerator.state.deepspeed_plugin is not None:
        args.gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ]
    # New Code #
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    # 为兼容iterable数据集，scheduler不进行学习率衰退
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
        )
    else:
        # lr_scheduler = DummyScheduler(
        #     optimizer,
        #     warmup_num_steps=args.num_warmup_steps
        # )
        lr_scheduler=None


    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    # Train!
    total_batch_size = args.per_device_train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps


    logger.info("***** Running training *****")
    logger.info("Num Epochs = {}".format(args.num_train_epochs))
    logger.info(
        "Instantaneous batch size per device = {}".format(args.per_device_train_batch_size))
    logger.info(
        "Total train batch size (w. parallel, distributed & accumulation) = {}".format(total_batch_size))
    logger.info(
        "Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    completed_steps = 0
    best_metric = None
    perplexity = None

    losses = []
    for epoch in tqdm(range(args.num_train_epochs), disable=not accelerator.is_local_main_process):
        if args.max_steps is not None and completed_steps > args.max_steps:
            break
        model.train()

        for step, batch in enumerate(train_dataloader):
            if os.path.exists(os.path.join(real_path,'..','..','..','flag.txt')):
                break
            outputs = model(**batch)

            loss = outputs.loss
            losses.append(loss.item())
            # We keep track of the loss at each epoch
            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                if (completed_steps + 1) % args.steps_per_print == 0:
                    accelerator.print(step,"loss:", loss.item())
                    os.makedirs(args.output_dir, exist_ok=True)
                    with open(args.output_dir+'/log.txt','a',encoding='utf8') as f:
                        f.write(f'epochs: {epoch}, batch: {completed_steps}, loss: {np.mean(losses)}\n')
                    losses = []
                if args.save_steps is not None and (completed_steps + 1) % args.save_steps == 0:
                    best_metric, perplexity = save_model(args, model, eval_dataloader, accelerator, epoch, best_metric, perplexity, tokenizer,True)
        if os.path.exists(os.path.join(real_path,'..','..','flag.txt')):

            best_metric, perplexity = save_model(args, model, eval_dataloader, accelerator, epoch, best_metric,
                                                 perplexity, tokenizer, False)
            print('Stopped')
            break
        best_metric, perplexity = save_model(args, model, eval_dataloader, accelerator, epoch, best_metric, perplexity, tokenizer,False)



def save_model(args, model, eval_dataloader, accelerator, epoch, best_metric, perplexity, tokenizer,to_output):
    if to_output:
        real_path = os.path.split(os.path.realpath(__file__))[0]
        name = os.path.split(args.output_dir)[-1]
        output_path = os.path.join(real_path, "..", "..", "..", "output", "LLM",name)

        if eval_dataloader is not None:
            perplexity, eval_loss = evaluate(
                args, model, eval_dataloader, accelerator)
            logger.info(
                "epoch {}: perplexity: {} eval_loss: {}".format(epoch, perplexity, eval_loss))
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        if accelerator.is_main_process:
            state = unwrapped_model.state_dict()
            unwrapped_model.save_pretrained(
                output_path,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state,
            )
            if not args.use_lora:
                tokenizer.save_pretrained(output_path)
                copy_custom_files(args.model_path, output_path)
        accelerator.wait_for_everyone()

    else:

        # New Code #
        # Tracks the best checkpoint and best metric
        if best_metric is None or best_metric > perplexity:
            if eval_dataloader is not None:
                best_metric = perplexity
                accelerator.print(
                    "New best metric: {} at epoch {}".format(best_metric, epoch))
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            if accelerator.is_main_process:
                state = unwrapped_model.state_dict()
                unwrapped_model.save_pretrained(
                    args.output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=state,
                )
                if not args.use_lora:
                    tokenizer.save_pretrained(args.output_dir)
                    copy_custom_files(args.model_path, args.output_dir)
            accelerator.wait_for_everyone()
    return best_metric, perplexity


if __name__ == "__main__":
    main()
