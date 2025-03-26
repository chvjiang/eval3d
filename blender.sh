export PATH=$PATH:/fs-computility/mllm/chuziyang/blender-3.2.2-linux-x64/

# n=10
t=('Trellis' '3DTopia-XL' 'hunyuan' 'crm' 'spard_glb' 'unique3d')
# let split=$n*8
# for index in {0..7}
# do
#     export CUDA_VISIBLE_DEVICES=${index}
#     for j in $(seq 1 ${n})
#     do
#         let num=$index*$n+$j-1
#         for file in ${t[@]}
#         do
#             blender -b -P b.py -- \
#                 --object_path ${file}.txt \
#                 --output_dir /mnt/hwfile/mllm/chuziyang/eval3d/light_v1/${file} \
#                 --engine CYCLES --split ${split} --index ${num} \
#                 --num_images 40 &
#         done
#     done
# done
# wait

for light in `ls lights/*.hdr`
do
    for file in ${t[@]}
    do
        python distribute.py --save_path /fs-computility/mllm/shared/chuziyang/eval3d/render \
            --method ${file} \
            --light ${light} &
        sleep 20
    done
    wait
done
wait