from utils import load_json_file

class Args:
    def __init__(self, config_path):
        # config 및 데이터 관련, pretrained model 로드
        self.config = load_json_file(config_path)
        
        self.train_path = self.config['train_path'] # jsonl 파일 경로
        self.val_path = self.config['val_path'] # jsonl 파일 경로
        self.max_seq_length = self.config['max_seq_length']
        self.pretrained_model_name = self.config['pretrained_model_name']

        # Lora 파라미터
        self.lora_r = self.config['lora_r']
        self.lora_alpha = self.config['lora_alpha']
        self.lora_target_modules = self.config['lora_target_modules']
        self.lora_dropout = self.config['lora_dropout']
        self.use_gradient_checkpointing = self.config['use_gradient_checkpointing'] # True 권장
        self.random_state = self.config['random_state']
        self.use_rslora = self.config['use_rslora'] # False 권장

        # training argument
        self.num_train_epochs = self.config['num_train_epochs']
        self.train_batch_size = self.config['train_batch_size']
        self.eval_batch_size = self.config['eval_batch_size']
        self.gradient_accumulation_steps = self.config['gradient_accumulation_steps']
        self.weight_decay = self.config['weight_decay']
        self.evaluation_strategy = self.config['evaluation_strategy']
        self.save_steps = self.config['save_steps']
        self.eval_steps = self.config['eval_steps']
        self.learning_rate = self.config['learning_rate']
        self.logging_steps = self.config['logging_steps']
        self.output_dir = self.config['output_dir']
        self.optim = self.config['optim']
        self.load_best_model_at_end = self.config['load_best_model_at_end']
        self.save_total_limit = self.config['save_total_limit']

        # trainer 관련
        self.packing = self.config['packing']
        self.model_use_cache = self.config['model_use_cache']
        