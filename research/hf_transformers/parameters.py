import torch


class HyperParams:
    def __init__(self):
        self.max_seq_length = 128
        self.do_lower_case = False
        self.n_gpu = torch.cuda.device_count()
        self.per_gpu_train_batch_size = 8
        self.per_gpu_val_batch_size = 8
        self.gradient_accumulation_steps = 1
        self.learning_rate = 1e-4
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8                # Epsilon for Adam optimizer
        self.max_grad_norm = 1.0                # Max gradient norm
        self.warmup_steps = 0
        self.num_workers = 2
        self.num_train_epochs = 3

        # Whether to use 16-bit (mixed) precision (through NVIDIA apex)
        self.fp16 = False
        # Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']
        # See details at https://nvidia.github.io/apex/amp.html
        self.fp16_opt_level = "O1"
