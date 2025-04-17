import os
import json
from tqdm import tqdm
import argparse

def find_files(root_dir, extensions = [], recursive = True):
    
    matched_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if extensions:
                if os.path.splitext(file)[1].lower() in extensions:
                    matched_files.append(os.path.join(root, file))
            else:
                matched_files.append(os.path.join(root, file))
        if not recursive:
            break
    return matched_files

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="directory to objects")
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir).rstrip(os.sep)
    obj_paths = find_files(root_dir=args.data_dir, extensions=['.obj','.glb'])

    paths_dict = {}
    for obj_path in tqdm(obj_paths):
        paths_dict[obj_path] = 0    # 0 means the obj is not rendered
    
    name = os.path.basename(args.data_dir)
    os.makedirs('data/json', exist_ok=True)
    with open(f'data/json/{name}.json', 'w') as f:
        json.dump(paths_dict, f, indent=2)