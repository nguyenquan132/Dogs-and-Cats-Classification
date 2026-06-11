import cv2
import torch
import numpy as np
import argparse
from .utils import val_transforms, set_seed

def inference(model, image, device, id2label, seed=42):
    set_seed(seed=seed)
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(
                f"Cannot read image: {image}"
            )
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError(...)
    
    transformed = val_transforms(image=img)
    pixel_values = transformed['image'].unsqueeze(0) 
    
    pixel_values = pixel_values.to(device)
    model = model.to(device)
    model.eval()
    
    # Inference
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        pred_class_id = probabilities.argmax(dim=1).item()
        confidence = probabilities[0][pred_class_id].item()
    
    predicted_label = id2label[pred_class_id]
    
    return predicted_label, confidence

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Cat vs Dog Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    image_path = args.image_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label2id = {'cat': 0, 'dog': 1}
    id2label = {v: k for k, v in label2id.items()}
    print(inference(best_model_path='model/best-model-transfer-learning.ckpt', image_path=image_path,
            device=device, label2id=label2id, id2label=id2label))