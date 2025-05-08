from models.simple_tokenizer import SimpleTokenizer
from models.viclip import ViCLIP
import torch
import numpy as np
import cv2

def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

def frames2tensor(vid_list, fnum=8, target_size=(224, 224)):
    assert(len(vid_list) >= fnum)
    v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    def normalize(data):
        return (data/255.0-v_mean)/v_std
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).float()
    return vid_tube

def viclip_image_score(video, image, model):
    assert isinstance(image, str), "Only 1 text is supported for reference" 
    model.eval()
    video = cv2.VideoCapture(video)
    frames = [x for x in _frame_from_video(video)]
    frames_tensor = frames2tensor(frames)
    image_tensor = frames2tensor([cv2.imread(image)], fnum=1)
    with torch.no_grad():
        video_feat = model.get_vid_features(frames_tensor)
        image_feat = model.get_vid_features(image_tensor)
        score = video_feat @ image_feat.T
    return score

def viclip_text_score(video, text, model):
    assert isinstance(text, str), "Only 1 text is supported for reference" 
    model.eval()
    video = cv2.VideoCapture(video)
    frames = [x for x in _frame_from_video(video)]
    frames_tensor = frames2tensor(frames)
    with torch.no_grad():
        video_feat = model.get_vid_features(frames_tensor)
        text_feat = model.get_text_features(text)
        score = video_feat @ text_feat.T
    return score

if __name__=='__main__':
    model_path = "/fs-computility/mllm/shared/chuziyang/model/ViCLIP/ViClip-InternVid-10M-FLT.pth"
    tokenizer_path = "/fs-computility/mllm/shared/chuziyang/model/ViCLIP/bpe_simple_vocab_16e6.txt.gz"
    viclip = ViCLIP(tokenizer_path=tokenizer_path, pretrain=model_path)

    score = viclip_image_score(video='/fs-computility/mllm/chuziyang/eval3d/data/train/albedo/hunyuan/0.mp4',
                               image='/fs-computility/mllm/chuziyang/eval3d/models/different_method.png',
                               model=viclip)
    print(score)
