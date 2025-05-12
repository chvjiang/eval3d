import datetime
import logging
import time
from os.path import join
import copy
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset

from dataset.serialize import local_broadcast_process_authkey
from dataset import MetaLoader, create_dataset, create_loader, create_sampler, create_stateful_sampler
from models import *
from utils.retrieval_utils import evaluation_wrapper
from utils.shared_utils import get_media_types, setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)


def setup_dataloaders(config, mode="pt"):
    logger.info(f"Creating dataset for {mode}")
    
    test_datasets = create_dataset(f"{mode}_eval", config)
    media_types = get_media_types(test_datasets)

    test_loaders = create_loader(
        test_datasets,
        [None] * len(test_datasets),
        batch_size=[config.inputs.batch_size_test[d] for d in media_types],
        num_workers=[config.num_workers] * len(test_datasets),
        is_trains=[False] * len(test_datasets),
        collate_fns=[None] * len(test_datasets),
    )
    return test_loaders


def main(config):
    metrics = ['Detail','Color','Light','Material']
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    
    test_loaders = setup_dataloaders(config, mode=config.mode)
    media_types = get_media_types(test_loaders)
    test_loader = MetaLoader(name2loader=dict(list(zip(media_types, test_loaders))))

    metric_logger = MetricLogger(delimiter="  ")
    for metric in metrics:
        metric_logger.add_meter(metric, SmoothedValue(window=1, fmt="{value:1.1f}"))
    header = f"Test sample: "
    iterator = metric_logger.log_every(test_loader, config.log_freq, header)
    
    model_cls = eval(config.model.get('model_cls', 'InternVideo2_CLIP'))
    logger.info("Creating model")
    config = copy.deepcopy(config)
    model = model_cls(config=config, is_pretrain=False)
    model = model.to(torch.device(config.device))
    if config.get('use_bf16', True):
        logger.info("Change to bfloat16 for model")
        model = model.to(torch.bfloat16)
        data_type = torch.bfloat16
    else:
        logger.info("Change to float16 for model")
        model = model.half()
        data_type = torch.float16

    tokenizer = model.tokenizer
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    start_time = time.time()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=config.use_half_precision, dtype=data_type):
            for _, (_, (image, text, idx, prompt, is_text_prompt, score)) in enumerate(iterator):
                print(image.shape)
                image = image.to(device, non_blocking=True)
                idx = idx.to(device, non_blocking=True)
                text_input = tokenizer(text).to(device)

                # predicts = model.generate(image, text_input, idx=idx, prompt=prompt, is_text_prompt=is_text_prompt)
                # for i, metric in enumerate(metrics):
                #     metric_logger.update(**{f"{metric}":predicts[0][i]-score[0][i].to(device)})

                sim = model.sim(image, text_input, idx=idx, prompt=prompt, is_text_prompt=is_text_prompt)[0]
                import matplotlib.pyplot as plt
                import seaborn as sns
                plt.figure(figsize=(8, 8))
                sns.heatmap(sim.detach().cpu().float().numpy(),
                            cmap="Blues",
                            annot=True,        # 显示数值
                            linewidths=.5,     # 单元格边框线宽
                            cbar=True)         # 显示颜色条
                plt.savefig('different_method.png')
                break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Inference time {total_time_str}")
    logger.info(f"Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    local_broadcast_process_authkey()
    main(cfg)
