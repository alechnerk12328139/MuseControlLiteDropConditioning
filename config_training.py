def get_config():
    return {
        # Load files and checkpoints

        "condition_type": ["drop",], #"melody", "rhythm", "dynamics", "audio", "drop"

        "meta_data_path": "./test_condition_s4.json",

        "audio_data_dir": "/var/home/drops-47s/house/",

        "audio_codec_root": "/var/home/drops-47s/house/",
        
        "drop_label_file": "./jamendo_labels.csv",

        "output_dir": "./checkpoints/stable_audio_drop_only_run03/",

        "transformer_ckpt": "./checkpoints/stable_audio_drop_only_run02/checkpoint-4000/model_1.safetensors", #"./checkpoints/woSDD-all/model_3.safetensors",

        "extractor_ckpt": {
             #"dynamics": "./checkpoints/stable_audio_all_condition_run11/checkpoint-11500/model_1.safetensors",
             #"melody": "./checkpoints/stable_audio_all_condition_run11/checkpoint-11500/model.safetensors",
             #"rhythm": "./checkpoints/stable_audio_all_condition_run11/checkpoint-11500/model_2.safetensors",
             "drop": "./checkpoints/stable_audio_drop_only_run02/checkpoint-4000/model.safetensors",
        },
        
        "train_melody_extractor": False,
        "train_dynamics_extractor": False,
        "train_rhythm_extractor": False,
        "train_drop_extractor": True,

        "wand_run_name": "test_drop_only",
        
        "start_step": 4000,

        # training hyperparameters
        "GPU_id" : "0",

        "train_batch_size": 3,

        "learning_rate": 1e-4,

        "attn_processor_type": "rotary", # "rotary", "rotary_conv_in", "absolute" 

        "gradient_accumulation_steps": 42,

        "max_train_steps": 200000,

        "num_train_epochs": 20,

        "dataloader_num_workers": 16,

        "mixed_precision": "fp16", #["no", "fp16", "bf16"]

        "apadapter": True,

        "lr_scheduler": "constant", # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'

        "weight_decay": 1e-2,

        #config for validation
        "validation_num": 500,

        "test_num": 5,

        "ap_scale": 1.0,

        "guidance_scale_text": 7.0,

        "guidance_scale_con": 1.5, # The separated guidance for both Musical attribute and audio conditions. Note that if guidance scale is too large, the audio quality will be bad. Values between 0.5~2.0 is recommended.

        "checkpointing_steps": 500,

        "validation_steps": 500,

        "denoise_step": 50,

        "log_first": True,

        "sigma_min": 0.3,

        "sigma_max": 500,
    }