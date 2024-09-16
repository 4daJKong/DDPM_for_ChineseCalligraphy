from transformers import BertModel, AutoTokenizer
from diffusers import UNet2DModel, UNet2DConditionModel, AutoencoderKL


def load_unet2duncond():
    model = UNet2DModel(
        sample_size=64,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model

def load_vae(model_path):
    model = AutoencoderKL.from_pretrained(model_path)
    return model

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    return (tokenizer, model)

def load_unet(image_size):
    '''image input with text, look at `forward` func'''
    model = UNet2DConditionModel(
        sample_size=image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        
        # encoder_hid_dim = 768,
        cross_attention_dim = 768,

        block_out_channels=(320, 640, 1280, 1280),  # the number of output channels for each UNet block
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", 
            "CrossAttnUpBlock2D", 
            "CrossAttnUpBlock2D", 
            "CrossAttnUpBlock2D",
        ),
    )

    '''only use image as input'''
    model_unet2d = UNet2DModel(
        sample_size=image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        # block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        block_out_channels=(320, 640, 1280, 1280),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            # "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            # "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            # "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            # "UpBlock2D",
        ),
    )
    return model



