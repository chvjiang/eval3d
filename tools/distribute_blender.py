import glob
import json
import multiprocessing
import shutil
import subprocess
import time
import os
import logging
import argparse

VC = ['dreamfusion','grm','grm_sv3d','instantmesh','latent-nerf','magic123','magic3d','mvdream','openlrm','sjc','syncdreamer','sz123','wonder3d','z123']
R = {
    'crm':                  [90, 0, -90],
    'wonder3d':             [90, 0, -90],
    'dreamfusion':          [0, 0, 0],
    'latent-nerf':          [0, 0, 0],
    'magic123':             [0, 0, 0],
    'magic3d':              [0, 0, 0],
    'mvdream':              [0, 0, 0],
    'syncdreamer':          [0, 0, 0],
    'sz123':                [0, 0, 0],
    'z123':                 [0, 0, 0],
    'luciddreamer_glb':     [-90, 0, 0],
    'triplanegaussian_glb': [-90, 0, 0],
    'triposr':              [-90, 0, 0],
    'spard_glb':            [0, 0, -90],
    'grm_sv3d':             [0, 0, 180],
    'instantmesh':          [0, 180, 180],
}

def worker(queue, count, gpu):
    while True:
        item = queue.get()
        if item is None:
            break
            
        obj_path = item['object_path']
        save_path = item['save_path']
        light_path = item['light_path']
        render_mode = item['render_mode']
        method = item['method'].lower()

        os.makedirs(save_path, exist_ok = True)
            
        if gpu != -1:
            command = (f" CUDA_VISIBLE_DEVICES={gpu} ")
            # command = (f" ")
        else:
            command = (f" GOMP_CPU_AFFINITY='0-63' OMP_NUM_THREADS=64 OMP_SCHEDULE=STATIC OMP_PROC_BIND=CLOSE ")

        command += (
            f" blender -b -P ./data/render_blender.py --"
            f" --object_path {obj_path}"
            f" --output_dir {save_path}"
            f" --mode {render_mode}"
            f" --num_images 40"
        )
        if light_path.endswith('.hdr'):
            command += f" --light_path {light_path}"
            command += f" --bg"
        else:
            command += f" --light_type {light_path}"
        if method in VC:
            command += f" --vc"
        if gpu == -1:
            command += f" --cpu"
        if method in R.keys():
            x, y, z = R[method]
            command += (
                f" --rotation_x {x}"
                f" --rotation_y {y}"
                f" --rotation_z {z}"
            )
        else:
            command += f" --rotation_z 90"

        subprocess.run(command, shell=True)

        logging.info(f'rendering:\t{obj_path}')

        with count.get_lock():
            count.value += 1

        queue.task_done()
    queue.task_done()
    

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--worker', type=int, default=4)
    parser.add_argument('--json', type=str, required=True)
    parser.add_argument('--light', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['light','object'])
    parser.add_argument('--save_path', type=str, default='./output')
    args = parser.parse_args()
    
    args.method = os.path.splitext(os.path.basename(args.json))[0]
    os.makedirs('./log', exist_ok=True)
    t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    logging.basicConfig(filename=f'./log/{args.method}-{t}.log', level=logging.INFO)

    use_gpus = ['0', '1', '2', '3', '4', '5', '6', '7']
    use_gpus = use_gpus[:args.gpus]
    
    if not use_gpus:
        use_gpus = [-1]
    workers_per_gpu = args.worker
    
    with open(args.json, 'r') as f:
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

    object_paths_list = list(object_paths.keys())
    object_paths_list.sort()

    os.makedirs(args.save_path, exist_ok=True)
    light_name = os.path.splitext(os.path.basename(args.light))[0]
    save_path = os.path.join(args.save_path, f'light_{light_name}')
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, args.method)

    for object_path in object_paths_list:
            
        queue_item = {}
        queue_item['object_path'] = object_path
        queue_item['save_path'] = save_path
        queue_item['light_path'] = args.light
        queue_item['render_mode'] = args.mode
        queue_item['method'] = args.method
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