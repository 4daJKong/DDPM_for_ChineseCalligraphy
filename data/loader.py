import os
import json
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModelForMaskedLM

import pdb

class ChCalliDataset(Dataset):

    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        with open(os.path.join(dataset_dir,'config.json'), 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        self.file_to_text = {
            item['file']: f"{item['char']}字 {item['script']}书 {item['style']}"
            for item in self.annotations
        }
        self.image_files = [f for f in os.listdir(os.path.join(dataset_dir, 'imgs')) if f in self.file_to_text]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, 'imgs', self.image_files[idx])

        image = Image.open(img_path).convert("RGB")

        # image = Image.open(img_path).convert("L")    
        # image = image.point(lambda p: 0 if p < 128 else 1)

        text = self.file_to_text[self.image_files[idx]]

        if self.transform:
            image = self.transform(image)
        return {"image": image, "text": text}

def get_dataset(dataset_path):
    dataset = load_dataset("imagefolder", data_dir=dataset_path, split="train")
    # dataset = load_dataset("imagefolder", data_dir=dataset_path, split=datasets.Split.TRAIN)
    # train_test_split = dataset.train_test_split(test_size=0.01)  # dataset['train']
    return dataset




def get_dataset_old(dataset_path, image_size):
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    dataset = ChCalliDataset(dataset_dir=dataset_path, transform=preprocess)

    # train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # for step, batch in enumerate(train_dataloader):
    #     clean_images = batch["images"]
    #     clean_text = batch["texts"]
    #     print(clean_images.shape, clean_text)
    #     pdb.set_trace()
    # sample_image = dataset[0]["images"].unsqueeze(0)
    # pdb.set_trace()

    return dataset




def get_dataset_butterfly(dataset_path, image_size):
    '''dataset[0]  // dataset.info // '''
    dataset = load_dataset(dataset_path, split="train")
    # fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    # for i, image in enumerate(dataset[:4]["image"]):
    #     axs[i].imshow(image)
    #     axs[i].set_axis_off()

    # fig.savefig('my_figure.png') 
    # fig_1 = dataset[4]["image"]
    # fig_1.save('my_figure_5.png')
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    '''canbe work'''
    # tensor_image = dataset[1]["images"]

    # transform_img = transforms.ToPILImage()
    # image = transform_img(tensor_image)
    # image.save('output_image.png')
    # pdb.set_trace()
    '''not work:'''
    # tensor_image = (tensor_image + 1) / 2
    # tensor_image = tensor_image * 255
    # numpy_image = tensor_image.numpy().astype(np.uint8)
    # numpy_image = tensor_image.permute(1, 2, 0).numpy()
    # image = Image.fromarray(numpy_image)
    # # image = Image.fromarray(numpy_image.squeeze(), mode='L')
    # image.save('output_image.png')
    # pdb.set_trace()
    return dataset




if __name__ == "__main__":
    pass
    get_dataset("/app/dpm_prj/dataset/calligraphy", image_size=64)
