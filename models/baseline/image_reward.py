import ImageReward as RM
import torch

def image_reward_score(image, text, model):
    assert isinstance(text, str), "Only 1 text is supported for reference"
    with torch.no_grad():
        score = model.score(prompt=text, image=image)
    return torch.tensor(score).mean()

if __name__=='__main__':

    model = RM.load("ImageReward-v1.0")

    score = image_reward_score(image=['/fs-computility/mllm/chuziyang/eval3d/models/different_method.png',
                                        '/fs-computility/mllm/chuziyang/eval3d/models/different_type.png'],
                                text='a diagram', model=model)
    print(score)