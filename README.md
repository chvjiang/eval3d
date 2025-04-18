# eval3d
We have provided a useful toolkit for 3D objects rendering, including HDRI application, lighting rotation, camera rotation, and so on.
## Quick Start
### Blender tools
#### Install Blender
You first need to download and install blender in your environment.
```
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
tar -xf blender-3.2.2-linux-x64.tar.xz
rm blender-3.2.2-linux-x64.tar.xz
export PATH=$PATH:path_to_blender/blender-3.2.2-linux-x64/
```
#### Single render
After installation, you can use ```_blender.py``` to render a single object. The parameters can be adjusted for your demand.
```
blender -b -P _blender.py -- \
    --object_path example.glb \
    --output_dir ./output \
    --light_path light.hdr \
    --engine CYCLES \
    --mode object \
    --num_images 40 \
```
#### Multiprocessing
We also support distributed rendering process for large quantities of objects.
```
python distribute.py --save_path ./output \
    --method your_files \
    --light light.hdr \
    --mode object --gpus 2 --worker 4
```
where path to objects should be stored in ```data/your_files.json```
### Kiui tools
#### Install Kiui
Besides Blender, kiuitools are also adopted in this toolkit.
```
pip install kiui
pip install git+https://github.com/NVlabs/nvdiffrast.git#egg=nvdiffrast
```