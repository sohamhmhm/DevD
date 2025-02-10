from transformers import CLIPProcessor, CLIPModel, T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
import torch

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained("saved_model").to(device)
tokenizer = T5Tokenizer.from_pretrained("saved_model")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generate_html(image_path):
    image = Image.open(image_path).resize((256, 256))
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(inputs["pixel_values"])
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Run inference
image_path = "test_image.png"
print(generate_html(image_path))
