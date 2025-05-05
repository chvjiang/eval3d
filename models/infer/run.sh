export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=1
NUM_CPU=8

torchrun --nnodes=${NNODE} \
--nproc_per_node=${NUM_GPUS} \
--rdzv_backend=c10d \
$(dirname $0)/inference.py \
$(dirname $0)/config.py
