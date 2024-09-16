

import os 
import torch
from PIL import Image

from dataclasses import dataclass

from data.loader import get_dataset
from model.loader import load_unet, load_tokenizer, load_vae

from workflow import train_loop

'''not sure if use...'''
from transformers import ( 
    CLIPTokenizer, #tokenizer
    CLIPTextModel #text_encoder
    )
from diffusers import (
    DDPMScheduler, #noise_scheduler
    AutoencoderKL, #vae
    StableDiffusionPipeline,
    PNDMScheduler,
    )


import pdb
#https://huggingface.co/docs/diffusers/en/tutorials/basic_training

@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 8  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 8
    learning_rate = 1e-5
    lr_warmup_steps = 500
    lr_scheduler = "constant_with_warmup" #'Choose between ["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"]'
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "/app/dpm_prj/output/calli-v5"  # the model name locally and on the HF Hub
    bert_dir = "/app/data/llm_factory/model/bert-base-chinese"
    vae_dir = "/app/data/llm_factory/model/stable-diffusion-v1-4/vae"
    # dataset_name = "/app/dpm_prj/dataset/smithsonian_butterflies_subset"
    dataset_name = "/app/dpm_prj/dataset/calligraphy"
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    # hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    # hub_private_repo = False
    max_grad_norm = 2
    report_to = "wandb" #Choose between ['all', 'aim', 'tensorboard', 'wandb', 'comet_ml', 'mlflow', 'clearml', 'dvclive']
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 42



def main():
    config = TrainingConfig() # print(config) #TrainingConfig()


    dataset = get_dataset(config.dataset_name)
    # pdb.set_trace()
    
    unet = load_unet(config.image_size)
    tokenizer_and_encoder = load_tokenizer(config.bert_dir)
    
    
    '''
    unet = UNet2DConditionModel.from_pretrained()

    noise_scheduler = DDPMScheduler.from_pretrained()
    tokenizer = CLIPTokenizer.from_pretrained()
    text_encoder = CLIPTextModel.from_pretrained()
    vae = AutoencoderKL.from_pretrained()
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    '''
    # noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise_scheduler = PNDMScheduler(num_train_timesteps=100)

    optimizer = torch.optim.AdamW(
        unet.parameters(), 
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )


    train_loop(config, dataset, unet, tokenizer_and_encoder, noise_scheduler, optimizer)


    # pdb.set_trace()

if __name__ == "__main__":

    main()
