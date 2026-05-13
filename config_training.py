def get_config():
    return {
        # Load files and checkpoints

        "condition_type": ["melody"], #["melody", "rhythm", "dynamics", "drop"], #"melody", "rhythm", "dynamics", "audio", "drop"

        "meta_data_path": "./mtg_full_47s_conditions/filtered_vocal_all_caption_with_conditions.json",

        "audio_data_dir": "D:/Datasets/drops-47s/house/",

        "audio_codec_root": "D:/Datasets/drops-47s/house/",

        "output_dir": "./checkpoints/stable_audio_all_condition",

        "transformer_ckpt": None, #"./checkpoints/stable_audio_melody_wo_SDD/checkpoint-5000/model_1.safetensors",

        "extractor_ckpt": {
             #"dynamics": "./checkpoints/110000_musical_44000_audio/model_1.safetensors",
             #"melody": "./checkpoints/stable_audio_melody_wo_SDD/checkpoint-5000/model.safetensors",
             #"rhythm": "./checkpoints/110000_musical_44000_audio/model_2.safetensors",
        },

        "wand_run_name": "test",

        # training hyperparameters
        "GPU_id" : "0",

        "train_batch_size": 1,

        "learning_rate": 1e-4,

        "attn_processor_type": "rotary", # "rotary", "rotary_conv_in", "absolute" 

        "gradient_accumulation_steps": 1,

        "max_train_steps": 200000,

        "num_train_epochs": 20,

        "dataloader_num_workers": 16,

        "mixed_precision": "fp16", #["no", "fp16", "bf16"]

        "apadapter": True,

        "lr_scheduler": "constant", # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'

        "weight_decay": 1e-2,

        #config for validation
        "validation_num": 30,

        "test_num": 5,

        "ap_scale": 1.0,

        "guidance_scale_text": 7.0,

        "guidance_scale_con": 1.5, # The separated guidance for both Musical attribute and audio conditions. Note that if guidance scale is too large, the audio quality will be bad. Values between 0.5~2.0 is recommended.

        "checkpointing_steps": 500,

        "validation_steps": 500,

        "denoise_step": 50,

        "log_first": False,

        "sigma_min": 0.3,

        "sigma_max": 500,
    }