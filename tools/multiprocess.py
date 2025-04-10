import glob
import json
import multiprocessing
import shutil
import subprocess
import time
import os

import argparse

def worker(queue, count):
    while True:
        item = queue.get()
        if item is None:
            break
            
        command = (
            f" python tools/convert.py "
            f" --input {item} "
            f" --output {item.replace('.obj','.glb')} "
            f" --resolution 1024 "
        )
        
        subprocess.run(command, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()
    queue.task_done()
    

def main(args):
    
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)
    
    workers = 32
    
    processes = []
    for _ in range(workers):
        process = multiprocessing.Process(
            target=worker, args=(queue, count)
        )
        process.daemon = True
        process.start()
        processes.append(process)

    object_paths_list = []
    
    data_path = args.data_path
    for i in os.listdir(data_path):
        if i.endswith('.obj'):
            object_paths_list.append(os.path.join(data_path, i))

    object_paths_list.sort()

    for object_path in object_paths_list:
        queue.put(object_path)
        
    for i in range(workers):
        queue.put(None)
        
    queue.join()
    for process in processes:
        process.join()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
