import os
from modules.model.Embedding.base import load_model, train_model, eval_model

real_path = os.path.realpath(__file__)

class embedding_base_model:
    def __init__(
        self,
        base_model_relative_path: str = '../../models/Embedding/',
        base_data_relative_path: str = '../../data/modeldata/Embedding/',
        base_output_relative_path: str = '../../outputs/Embedding/',
    ):
        self.base_model_relative_path = base_model_relative_path
        self.base_data_relative_path = base_data_relative_path
        self.base_output_relative_path = base_output_relative_path
        self.model = None
        
    def clear(self):
        if self.model:
            self.model.clear()
            self.model = None
        
    def load(self, model_name_or_path: str, embed_arch:str, device: str = None):
        model = load_model(model_name_or_path, embed_arch, device = device)
        self.model = model
        
    def reload(self, model_name_or_path: str, device: str = None):
        self.clear()
        model = load_model(model_name_or_path, device = device)
        self.model = model
    
    def train(
        self,
        train_file: str = '../../data/modeldata/Embedding/STS-B/STS-B.train.data',
        output_dir: str = '../../outputs/Embedding/X-STSB',
        eval_file: str= '../../data/modeldata/Embedding/STS-B/STS-B.valid.data',
        verbose: bool = True,
        batch_size: int = 32,
        num_epochs: int = 10,
        weight_decay: float = 0.01,
        seed: int = 42,
        warmup_ratio: float = 0.1,
        lr: float = 2e-5,
        eps: float = 1e-6,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        max_steps: int = -1,
        logging_epochs: int = 1,
        final_model=None
    ):
        for global_step, training_details in train_model(
            self.model,
            train_file,
            output_dir,
            eval_file = eval_file,
            verbose = verbose,
            batch_size = batch_size,
            num_epochs = num_epochs,
            weight_decay = weight_decay,
            seed = seed,
            warmup_ratio = warmup_ratio,
            lr = lr,
            eps = eps,
            gradient_accumulation_steps = gradient_accumulation_steps,
            max_grad_norm = max_grad_norm,
            max_steps = max_steps,
            final_model=final_model,
            logging_epochs = logging_epochs
        ):
            yield global_step, training_details

            

    def evaluate(
        self,
        eval_file: str= '../../data/modeldata/Embedding/STS-B/STS-B.valid.data',
        output_dir: str = '../../outputs/Embedding/X-STSB',
        verbose: bool = True,
        batch_size: int = 16,
    ):
        result = eval_model(
            self.model,
            eval_file,
            output_dir = output_dir,
            verbose = verbose,
            batch_size = batch_size,
        )
        
        return result
