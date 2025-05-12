from clip import clip_image_score, clip_text_score
from viclip import viclip_image_score, viclip_text_score
from aesthetics import get_aesthetic_model, aesthetic_score
from image_reward import image_reward_score
from FID import calculate_fid_given_paths
import multiprocessing as mp
import random
import os
import json
import open_clip
import ImageReward as RM
from models.viclip import ViCLIP

class BaseMetrics:
    def __init__(self, device='cuda'):
        clip_path = "/fs-computility/mllm/shared/chuziyang/models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_model.safetensors"
        clip_name = 'ViT-H-14'
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_name, pretrained=clip_path)
        clip_tokenizer = open_clip.get_tokenizer(clip_name)
        self.clip_model, self.clip_preprocess, self.clip_tokenizer = clip_model, clip_preprocess, clip_tokenizer

        viclip_path = "/fs-computility/mllm/shared/chuziyang/model/ViCLIP/ViClip-InternVid-10M-FLT.pth"
        viclip_tokenizer_path = "/fs-computility/mllm/shared/chuziyang/model/ViCLIP/bpe_simple_vocab_16e6.txt.gz"
        self.viclip_model = ViCLIP(tokenizer_path=viclip_tokenizer_path, pretrain=viclip_path)

        image_reward_path = "/fs-computility/mllm/shared/chuziyang/model/ImageReward/ImageReward.pt"
        self.image_reward_model = RM.load(image_reward_path)

        aesthetic_path = "/fs-computility/mllm/chuziyang/eval3d/ckpts/aesthetic-predictor/sa_0_4_vit_l_14_linear.pth"
        aesthetic_name = 'ViT-L-14'
        self.aesthetic_model = get_aesthetic_model(model_path=aesthetic_path, clip_model=aesthetic_name)
        aesthetic_clip_model, _, aesthetic_preprocess = open_clip.create_model_and_transforms(aesthetic_name, pretrained='openai')
        self.aesthetic_clip_model, self.aesthetic_preprocess = aesthetic_clip_model, aesthetic_preprocess

        self.clip_model = self.clip_model.to(device)
        self.viclip_model = self.viclip_model.to(device)
        self.image_reward_model = self.image_reward_model.to(device)
        self.aesthetic_clip_model = self.aesthetic_clip_model.to(device)
        self.aesthetic_model = self.aesthetic_model.to(device)
        self.device = device

    def get_metrics(self, obj: dict):
        clip_score = []
        viclip_score = []
        aesthetics_score = []
        ImageReward_score = []

        for video in obj['video']:
            video = os.path.join("/fs-computility/mllm/chuziyang/eval3d/data", video)
            if os.path.isfile(obj['prompt']):
                viclip_score.append(viclip_image_score(video=video, 
                                                       image=obj['prompt'],
                                                       model=self.viclip_model))
            else:
                viclip_score.append(viclip_text_score(video=video, 
                                                      text=obj['prompt'], 
                                                      model=self.viclip_model))

        for images in obj['images']:
            images = [os.path.join(images, p) for p in os.listdir(images)]
            if os.path.isfile(obj['prompt']):
                clip_score.append(clip_image_score(image1=images, 
                                                   image2=obj['prompt'], 
                                                   model=self.clip_model, 
                                                   preprocess=self.clip_preprocess))
            else:
                clip_score.append(clip_text_score(image=images, 
                                                  text=obj['prompt'], 
                                                  model=self.clip_model, 
                                                  preprocess=self.clip_preprocess, 
                                                  tokenizer=self.clip_tokenizer))
                ImageReward_score.append(image_reward_score(image=images, 
                                                            text=obj['prompt'], 
                                                            model=self.image_reward_model))
            aesthetics_score.append(aesthetic_score(image=images, 
                                                    model=self.aesthetic_clip_model, 
                                                    amodel=self.aesthetic_model, 
                                                    preprocess=self.aesthetic_preprocess))
        order = [2, 0, 1, 3]
        obj['clip_score'] = [float(clip_score[i].cpu().numpy()) for i in order] if clip_score else []
        obj['viclip_score'] = [float(viclip_score[i].cpu().numpy()) for i in order] if viclip_score else []
        obj['aesthetic_score'] = [float(aesthetics_score[i].cpu().numpy()) for i in order] if aesthetics_score else []
        obj['image_reward'] = [float(ImageReward_score[i].cpu().numpy()) for i in order] if ImageReward_score else []
        return obj

def worker(input_queue, output_queue, worker_id):
    
    base_metrics = BaseMetrics()
    
    while True:
        obj = input_queue.get()
        if obj is None:
            print(f"worker {worker_id:2d} exits")
            break

        print('Processing:', obj['video'][0])        

        obj = base_metrics.get_metrics(obj)

        # output_queue.put(obj)

        temp = json.load(open('data/baseline_test_result.json'))
        with open('data/baseline_test_result.json','w') as f:
            json.dump(temp+[obj], f, indent=4)

if __name__=='__main__':
    mp.set_start_method('spawn')
    cpus = 2 #mp.cpu_count() // 2
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    processes = []
    for worker_id in range(cpus):
        process = mp.Process(
            target=worker, args=(input_queue, output_queue, worker_id)
        )
        process.daemon = True
        process.start()
        processes.append(process)
    
    objs = json.load(open('data/baseline_test.json'))

    with open('data/baseline_test_result.json','w') as f:
        json.dump([], f, indent=4)

    for obj in objs:
        input_queue.put(obj)

    for _ in range(cpus):
        input_queue.put(None)
    
    for process in processes:
        process.join()

    # results = []
    # while not output_queue.empty():
    #     results.append(output_queue.get())
    