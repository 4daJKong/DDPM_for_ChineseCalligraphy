import os
import math
import random

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from pathlib import Path
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger

from huggingface_hub import create_repo, upload_folder

from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.optimization import (
    get_scheduler, 
    get_cosine_schedule_with_warmup
    )


import pdb


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config, dataset, unet, tokenizer_and_encoder, noise_scheduler, optimizer):
    
    tokenizer = tokenizer_and_encoder[0]
    text_encoder = tokenizer_and_encoder[1]
    #preprocess dataset

    '''image tensorinize'''
    train_transforms = transforms.Compose(
        [
            transforms.Resize(config.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    

    '''text tokenizer'''
    def tokenize_captions(examples, caption_column='text', is_train=True):
        captions = []
        for caption in examples[caption_column]:
            captions.append(caption)   
        inputs = tokenizer(captions, max_length=12, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids

    '''image transform'''
    def preprocess_train(examples, image_column = 'image'):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples
    
    
    train_dataset = dataset['train'].with_transform(preprocess_train)
    

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config.train_batch_size,
    )
    
    # pdb.set_trace()     

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=config.lr_warmup_steps,
    #     num_training_steps=(len(train_dataloader) * config.num_epochs),
    # )
    '''
    Initialize accelerator
    '''
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # log_with=config.report_to,
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")
    '''
    Prepare everything
    There is no specific order to remember, you just need to unpack the
    objects in the same order you gave them to the prepare method.
    '''
    # vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    '''add from train_text_to_image.py'''
    #start
    weight_dtype = torch.bfloat16
    
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    config.max_train_steps = config.num_epochs * num_update_steps_per_epoch
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
 
    
    global_step = 0

    # pdb.set_trace()
    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )
    for epoch in range(config.num_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            with accelerator.accumulate(unet):
                '''
                # L967
                # without vae encode...not said in paper, 
                assuming latents==images
                channel=3 if images else 4
                '''
                latents = batch["pixel_values"].to(weight_dtype)
                
                '''
                # batch["pixel_values"].shape torch.Size([16, 3, 64, 64])
                # batch['input_ids'].shape torch.Size([16, 12])
                '''
                # pdb.set_trace()
                noise = torch.randn_like(latents)
                
                bsz = latents.shape[0]
                
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) #L997
                
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0] #L1001
                
                target = noise #L1009 
                
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            #accumulate end
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
        
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
        
        #out of `for step`
        if accelerator.is_main_process: #L1109
            pass

    #out of `for epoch`
    accelerator.wait_for_everyone()
    
    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        '''save_model'''
        unet_path = os.path.join(config.output_dir, "unet")
        os.makedirs(unet_path, exist_ok=True)
        unet.save_pretrained(unet_path)

    
    accelerator.end_training()

    # train_transforms
    # preprocess_train  
    # train_dataset   
    # train_dataloader = torch.utils.data.DataLoader(
        # train_dataset,       
    #endif

'''
    bert_tokenizer = bert_tokenizer_model[0]
    bert_model = bert_tokenizer_model[1]

    global_step = 0

   
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), 
            disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            clean_texts = batch["texts"]
            
            #embedding texts
            inputs_tokens = bert_tokenizer(clean_texts, padding='max_length', max_length=12, truncation=True, return_tensors="pt")
            hidden_states = bert_model(**inputs_tokens).last_hidden_state
            hidden_states = hidden_states.to(clean_images.device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps) #torch.Size([16, 3, 64, 64])

            with accelerator.accumulate(unet):
                # Predict the noise residual
                # 这里添加hidden_state
                noise_pred = unet(noisy_images, timesteps, encoder_hidden_states=hidden_states, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the unet
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(unet), scheduler=noise_scheduler)

            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            #     evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)
'''



# if __name__ == "__main__":
  

#     from transformers import BertModel, AutoTokenizer

#     model_path = "/app/data/llm_factory/model/bert-base-chinese"
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     text_encoder = BertModel.from_pretrained(model_path)
   
#     text = ['狩字 行书 王铎', '舆字 隶书 何绍基', '焖字 楷书 智永', '戕字 行书 集字圣教序', '钉字 隶书 金农', '寮字 隶书 邓石如', '腰字 行书 傅山', '校字 楷书 赵孟俯三门记', '碣字 楷书 赵孟俯三门记', '痫字 楷书 魏碑', '添字 楷书 柳公权', '孓字 行书 米芾', '叱字 行书 王壮为', '当字 楷书 沈尹默', '淄字 行书 郑板桥', '国字 行书 集字圣教序']

#     max_length = 12
#     inputs = tokenizer(text, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")

#     encoder_hidden_states = text_encoder(inputs['input_ids'], return_dict=False)[0]
#     '''encoder_hidden_states.shape: torch.Size([16, 12, 768])'''
#     pdb.set_trace()
