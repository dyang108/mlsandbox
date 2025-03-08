from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset

# Load Stable Diffusion 2.1
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "stabilityai/stable-diffusion-2-1-base"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to(device)  # Move model to M1 GPU

# Apply LoRA for Fine-Tuning
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1
)
pipe.unet = get_peft_model(pipe.unet, lora_config)

# Load Dataset
dataset = load_dataset("imagefolder", data_dir="magazine_dataset")

# Set Optimizer
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)

# Training Loop
for epoch in range(3):  # Number of epochs
    for batch in dataset["train"]:
        images = batch["image"].to(device)
        captions = batch["text"] if "text" in batch else None

        loss = pipe(images, captions).loss  # LoRA computes loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned LoRA model
pipe.save_pretrained("sd2.1_magazine_lora")
print("LoRA Fine-Tuning Complete! Model saved to sd2.1_magazine_lora")
