import open_clip
from PIL import Image
import torch

def clip_image_score(image1, image2, model, preprocess, device='cuda'):
    assert isinstance(image2, str), "Only 1 image is supported for reference"
    model.eval()
    image1 = [image1] if isinstance(image1, str) else image1
    input1 = torch.stack([preprocess(Image.open(img)) for img in image1], dim=0)
    input2 = preprocess(Image.open(image2)).unsqueeze(0)
    with torch.no_grad():
        features1 = model.encode_image(input1.to(device))
        features2 = model.encode_image(input2.to(device))
        features1 /= features1.norm(dim=-1, keepdim=True)
        features2 /= features2.norm(dim=-1, keepdim=True)
        score = (features1 @ features2.T).mean()
    return score

def clip_text_score(image, text, model, preprocess, tokenizer, device='cuda'):
    assert isinstance(text, str), "Only 1 text is supported for reference"
    model.eval()
    image = [image] if isinstance(image, str) else image
    input1 = torch.stack([preprocess(Image.open(img)) for img in image], dim=0)
    input2 = tokenizer(text)
    with torch.no_grad():
        features1 = model.encode_image(input1.to(device))
        features2 = model.encode_text(input2.to(device))
        features1 /= features1.norm(dim=-1, keepdim=True)
        features2 /= features2.norm(dim=-1, keepdim=True)
        score = (features1 @ features2.T).mean()
    return score

if __name__=='__main__':

    model_path = "/fs-computility/mllm/shared/chuziyang/models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_model.safetensors"
    model_name = 'ViT-H-14'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_path)
    tokenizer = open_clip.get_tokenizer(model_name)
    
    score = clip_text_score(image=['/fs-computility/mllm/chuziyang/eval3d/models/different_method.png',
                                   '/fs-computility/mllm/chuziyang/eval3d/models/different_type.png'],
                            text='a diagram', model=model, preprocess=preprocess, tokenizer=tokenizer)
    print(score)
