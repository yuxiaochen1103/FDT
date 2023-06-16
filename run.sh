echo "start launch tasks ..."

echo $@
echo
GPU_NUM=8
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR='0.0.0.0'
MASTER_PORT='29500'

torchrun \
    --nproc_per_node ${GPU_NUM} \
    --nnodes ${NODE_NUM} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    $@ || exit 1

exit 1

echo "finish training ..."

sleep 10