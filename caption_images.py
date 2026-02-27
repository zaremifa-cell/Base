import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
from tqdm import tqdm

def main():
    root_dir = Path("/Users/zlatkoanastasov/Documents/Base/pinterest/zanastasov")
    
    print("Loading BLIP model...")
    # Initialize processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Use MPS (Metal Performance Shaders) on Mac if available, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    print(f"Scanning directory: {root_dir}")
    image_files = []
    
    # Recursively find all images
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in image_extensions:
            # Check if text file already exists
            txt_path = p.with_suffix(".txt")
            if not txt_path.exists():
                image_files.append(p)
                
    print(f"Found {len(image_files)} images needing captions.")
    
    if len(image_files) == 0:
        print("All images are already captioned.")
        return

    print("Generating captions...")
    for img_path in tqdm(image_files):
        try:
            # Load the image
            raw_image = Image.open(img_path).convert('RGB')
            
            # Unconditional image captioning
            inputs = processor(raw_image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            # Save the caption next to the image with the same name, but .txt extension
            txt_path = img_path.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption.strip() + "\n")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    main()
