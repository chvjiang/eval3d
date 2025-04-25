import os
import json
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
    parser.add_argument('--output_dir', type=str, default='./data/json', help="save directory")
    parser.add_argument('--split', type=int, default=1, help="split to multiple jsons")
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir).rstrip(os.sep)
    obj_paths = find_files(root_dir=args.data_dir, extensions=['.obj','.glb'])

    print(f'find {len(obj_paths):^5} object files in {args.data_dir}')
    
    name = os.path.basename(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.split == 1:
        paths_dict = {}
        for obj_path in obj_paths:
            paths_dict[obj_path] = 0    # 0 means the obj is not rendered
        with open(os.path.join(args.output_dir,f'{name}.json'), 'w') as f:
            json.dump(paths_dict, f, indent=2)
    else:
        step = len(obj_paths)/args.split
        for i in range(args.split):
            split_data = {}
            for obj_path in obj_paths[int(i*step):int((i+1)*step)]:
                split_data[obj_path] = 0    # 0 means the obj is not rendered
            with open(os.path.join(args.output_dir,f'{name}_split_{i}.json'), 'w') as f:
                json.dump(split_data, f, indent=2)