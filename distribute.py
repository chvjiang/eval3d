import glob
import json
import multiprocessing
import shutil
import subprocess
import time
import os

import argparse

def worker(queue, count, gpu):
    while True:
        item = queue.get()
        if item is None:
            break
            
        obj_path = item['object_path']
        save_path = item['save_path']
        light_path = item['light_path']
        render_mode = item['render_mode']

        os.makedirs(save_path, exist_ok = True)
            
        # Perform some operation on the item
        command = (
            # f"export DISPLAY=:0.{gpu} &&"
            f" GOMP_CPU_AFFINITY='0-47' OMP_NUM_THREADS=48 OMP_SCHEDULE=STATIC OMP_PROC_BIND=CLOSE "
            f" CUDA_VISIBLE_DEVICES={gpu} "
            # f" blender -b -P _blender.py --"
            f" blender -b -P _blender.py --"
            f" --object_path {obj_path}"
            f" --output_dir {save_path}"
            f" --light_path {light_path}"
            f" --mode {render_mode}"
            f" --num_images 40"
            f" --engine CYCLES"
            #f" > /dev/null 2>&1"
        )
        subprocess.run(command, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()
    queue.task_done()
    

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--method', type=str, required=True,
                        choices=['Trellis','3DTopia-XL','LN3Diff','crm','hunyuan','spard_glb','triposr','unique3d'])
    parser.add_argument('--light', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['light','object'])
    parser.add_argument('--save_path', type=str, default='./output')
    args = parser.parse_args()
    
    use_gpus = ['0', '1', '2', '3', '4', '5', '6', '7']
    use_gpus = use_gpus[:args.gpus]
    
    workers_per_gpu = 4
    
    with open(f'data/{args.method}.json') as f:
        object_paths = json.load(f)
    
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)
    
    processes = []
    for gpu_i in use_gpus:
        for _ in range(workers_per_gpu):
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i)
            )
            process.daemon = True
            process.start()
            processes.append(process)

    object_paths_list = list(object_paths.values())
    object_paths_list.sort()

    os.makedirs(args.save_path,exist_ok=True)
    light_name = args.light.split('/')[-1].split('.')[0]
    save_path = os.path.join(args.save_path,f'light_{light_name}')
    os.makedirs(save_path,exist_ok=True)
    save_path = os.path.join(save_path,args.method)

    for object_path in object_paths_list:
            
        queue_item = {}
        queue_item['object_path'] = object_path
        queue_item['save_path'] = save_path
        queue_item['light_path'] = args.light
        queue_item['render_mode'] = args.mode
        queue.put(queue_item)
        
    # Add sentinels to the queue to stop the worker processes
    for i in range(len(use_gpus) * workers_per_gpu):
        queue.put(None)
        
    # Wait for all tasks to be completed
    queue.join()
    for process in processes:
        process.join()
    
if __name__ == '__main__':
    main()