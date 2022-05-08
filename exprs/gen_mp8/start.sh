set -x
export PADDLE_WITH_GLOO=0
export FLAGS_call_stack_level=2
export FLAGS_convert_all_blocks=1
export FLAGS_allocator_strategy=auto_growth
export FLAGS_START_PORT=6000

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
rm -rf ./int8_bs${batch_size}
python -u -m paddle.distributed.fleet.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "./int8_bs${batch_size}" inf.py
#    --nranks 8 \
#    --batch_size ${batch_size}
