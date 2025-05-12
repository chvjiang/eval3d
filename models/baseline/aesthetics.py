import os
import torch
import torch.nn as nn
from urllib.request import urlretrieve
import torch
from PIL import Image
import open_clip

def get_aesthetic_model(model_path, clip_model="vit_l_14"):
    """load the aethetic model"""

    if not os.path.exists(model_path):
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, model_path)

    if clip_model == "vit_l_14" or clip_model == 'ViT-L-14':
        model = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        model = nn.Linear(512, 1)
    else:
        raise ValueError()
    
    s = torch.load(model_path)
    model.load_state_dict(s)
    model.eval()
    return model

def aesthetic_score(image, model, amodel, preprocess, device='cuda'):
    amodel.eval()
    model.eval()
    image = [image] if isinstance(image, str) else image
    image = torch.stack([preprocess(Image.open(img)) for img in image], dim=0)
    with torch.no_grad():
        image_features = model.encode_image(image.to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = amodel(image_features)
    return score.mean()

if __name__=='__main__':
    model_path = "/fs-computility/mllm/chuziyang/eval3d/ckpts/aesthetic-predictor/sa_0_4_vit_l_14_linear.pth"
    amodel = get_aesthetic_model(model_path=model_path, clip_model="vit_l_14")

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    score = aesthetic_score(image=['/fs-computility/mllm/chuziyang/eval3d/models/different_method.png',
                                   '/fs-computility/mllm/chuziyang/eval3d/models/different_type.png'],
                            model=model, amodel=amodel, preprocess=preprocess)
    print(score)
