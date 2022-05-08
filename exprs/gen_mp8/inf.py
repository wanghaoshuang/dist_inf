import paddle.distributed.fleet as fleet
import paddle
from paddle.fluid import core
import numpy as np
from paddle.fluid import profiler
import paddle.inference as paddle_infer
import time

paddle.enable_static()
fleet.init(is_collective=True)

use_trt = True
multi_gpu=True

if multi_gpu:
    path_prefix = "/root/paddlejob/workspace/env_run/whs/distributed-inference/ernie_inf/inference_model_gen_10b_mp8"
    model_path = path_prefix + "/rank_" + str(fleet.worker_index()) + "/step_0.pdmodel"
    param_path = path_prefix + "/rank_" + str(fleet.worker_index()) + "/step_0.pdiparams"
else:
    path_prefix = "/root/paddlejob/workspace/env_run/whs/distributed-inference/ernie_inf/inference_model_gen_10b_mp8"
    model_path = path_prefix + "/rank_0" + "/step_0.pdmodel"
    param_path = path_prefix + "/rank_0" + "/step_0.pdiparams"

generation = True
nranks = 8
bsz = 32
input_seq_len = 60
max_input_seq_len = 80
max_dec_len = 20
hidden_size = 4096
head_num = int(64 / nranks)
branch_head_num = int(12 / nranks)


if multi_gpu:
    current_endpoint = "127.0.0.1:600" + str(fleet.worker_index())
    if nranks == 1:
        trainer_endpoints = ["127.0.0.1:6000"]
    elif nranks == 2:
        trainer_endpoints = ["127.0.0.1:6000", "127.0.0.1:6001"]
    elif nranks == 4:
        trainer_endpoints = ["127.0.0.1:6000", "127.0.0.1:6001", "127.0.0.1:6002", "127.0.0.1:6003"]
    elif nranks == 8:
        trainer_endpoints = ["127.0.0.1:6000", "127.0.0.1:6001", "127.0.0.1:6002", "127.0.0.1:6003", "127.0.0.1:6004", "127.0.0.1:6005", "127.0.0.1:6006", "127.0.0.1:6007"]
    
    dist_config = core.DistConfig()
    dist_config.set_carrier_id("inference")
    dist_config.set_comm_init_config("./config.csv")
    dist_config.set_endpoints(trainer_endpoints, current_endpoint)
    dist_config.set_ranks(nranks, fleet.worker_index())
    dist_config.enable_dist_model(True)

analysis_config = core.AnalysisConfig()
analysis_config.enable_use_gpu(100, fleet.worker_index())
#analysis_config.enable_use_gpu(100, 0)
#analysis_config.switch_use_feed_fetch_ops(False)
analysis_config.set_model(model_path, param_path)

analysis_config.enable_memory_optim(False)
analysis_config.switch_ir_optim()

if multi_gpu:
    analysis_config.set_dist_config(dist_config)

#analysis_config.enable_profile()

if use_trt:
    analysis_config.enable_tensorrt_engine(workspace_size = 1 << 30,
                                  max_batch_size = bsz,
                                  min_subgraph_size = 10,
#                                  precision_mode=paddle_infer.PrecisionType.Half,
                                  precision_mode=paddle_infer.PrecisionType.Int8,
                                  use_static = False,
                                  use_calib_mode = False)
    min_input_shape = {
#            'fc_0.tmp_1': [1, 1, 16, 48],
#            'fc_1.tmp_1': [1, 1, 16, 48],
            'embedding_1.tmp_0': [1, input_seq_len, hidden_size],
            'embedding_4.tmp_0': [1, input_seq_len, hidden_size],
#            'unsqueeze2_0.tmp_0': [1, 1, 1, 1],
            'c_embedding_0.tmp_0': [bsz, input_seq_len, hidden_size],
            'embedding_2.tmp_0': [bsz, input_seq_len, hidden_size],
            'input_mask': [bsz, input_seq_len, input_seq_len]
    }
    max_input_shape = {
#        'fc_0.tmp_1': [bsz, input_seq_len, 16, 48],
#        'fc_1.tmp_1': [bsz, input_seq_len, 16, 48],
        'embedding_1.tmp_0': [bsz, input_seq_len, hidden_size],
        'embedding_4.tmp_0': [bsz, input_seq_len, hidden_size],
#        'unsqueeze2_0.tmp_0': [bsz, 1, input_seq_len, input_seq_len],
        'c_embedding_0.tmp_0': [bsz, input_seq_len, hidden_size],
        'embedding_2.tmp_0': [bsz, input_seq_len, hidden_size],
        'input_mask': [bsz, input_seq_len, input_seq_len]
    }
    opt_input_shape = {
#        'fc_0.tmp_1': [bsz, input_seq_len, 16, 48],
#        'fc_1.tmp_1': [bsz, input_seq_len, 16, 48],
        'embedding_1.tmp_0': [bsz, input_seq_len, hidden_size],
        'embedding_4.tmp_0': [bsz, input_seq_len, hidden_size],
#        'unsqueeze2_0.tmp_0': [bsz, 1, input_seq_len, input_seq_len],
        'c_embedding_0.tmp_0': [bsz, input_seq_len, hidden_size],
        'embedding_2.tmp_0': [bsz, input_seq_len, hidden_size],
        'input_mask': [bsz, input_seq_len, input_seq_len]
    }

    min_input_shape["array_read_3.tmp_0"] = [1]
    max_input_shape["array_read_3.tmp_0"] = [input_seq_len+1]
    opt_input_shape["array_read_3.tmp_0"] = [input_seq_len+1]
    min_input_shape["assign_31.tmp_0"] = [30000]
    max_input_shape["assign_31.tmp_0"] = [30000]
    opt_input_shape["assign_31.tmp_0"] = [30000]
    min_input_shape["c_embedding_1.tmp_0"] = [bsz, 1, hidden_size]
    max_input_shape["c_embedding_1.tmp_0"] = [bsz, 1, hidden_size]
    opt_input_shape["c_embedding_1.tmp_0"] = [bsz, 1, hidden_size]
    min_input_shape["embedding_5.tmp_0"] = [bsz, 1, hidden_size]
    max_input_shape["embedding_5.tmp_0"] = [bsz, 1, hidden_size]
    opt_input_shape["embedding_5.tmp_0"] = [bsz, 1, hidden_size]

    min_input_shape["embedding_6.tmp_0"] = [bsz, 1, hidden_size]
    max_input_shape["embedding_6.tmp_0"] = [bsz, 1, hidden_size]
    opt_input_shape["embedding_6.tmp_0"] = [bsz, 1, hidden_size]
    min_input_shape["embedding_9.tmp_0"] = [bsz, 1, hidden_size]
    max_input_shape["embedding_9.tmp_0"] = [bsz, 1, hidden_size]
    opt_input_shape["embedding_9.tmp_0"] = [bsz, 1, hidden_size]


    min_input_shape["stack_2.tmp_0"] = [bsz, head_num, 1, 1]
    max_input_shape["stack_2.tmp_0"] = [bsz, head_num, 1, max_input_seq_len]
    opt_input_shape["stack_2.tmp_0"] = [bsz, head_num, 1, max_input_seq_len]
    min_input_shape["stack_3.tmp_0"] = [bsz, branch_head_num, 1, 1]
    max_input_shape["stack_3.tmp_0"] = [bsz, branch_head_num, 1, max_input_seq_len]
    opt_input_shape["stack_3.tmp_0"] = [bsz, branch_head_num, 1, max_input_seq_len]
    min_input_shape["tmp_153"] = [30000]
    max_input_shape["tmp_153"] = [30000]
    opt_input_shape["tmp_153"] = [30000]

    min_input_shape["fill_constant_3.tmp_0"] = [1]
    max_input_shape["fill_constant_3.tmp_0"] = [1]
    opt_input_shape["fill_constant_3.tmp_0"] = [1]



    for i in range(768): # for KVCache
        var_ = "fill_constant_batch_size_like_"+str(i)+".tmp_0"
        min_input_shape[var_] = [bsz, head_num, 0, 64]
        max_input_shape[var_] = [bsz, head_num, max_input_seq_len, 64] 
        opt_input_shape[var_] = [bsz, head_num, max_input_seq_len, 64] 
    for i in range(768, 960): # for KVCache of branch
        var_ = "fill_constant_batch_size_like_"+str(i)+".tmp_0"
        min_input_shape[var_] = [bsz, branch_head_num, 0, 96]
        max_input_shape[var_] = [bsz, branch_head_num, max_input_seq_len, 96] 
        opt_input_shape[var_] = [bsz, branch_head_num, max_input_seq_len, 96]


    analysis_config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape, opt_input_shape)

print("--------------core.create_predictor--------------")
predictor = core.create_predictor(analysis_config)


lods = [list(range(bsz + 1)), list(range(bsz + 1))]
src_ids = np.ones((bsz, input_seq_len), dtype='int64')
pos_ids = np.ones((bsz, input_seq_len), dtype='int64')
input_mask =np.ones((bsz, input_seq_len, input_seq_len), dtype='float32')
tgt_ids = np.ones((bsz, 1), dtype='int64')
tgt_pos = np.ones((bsz, 1), dtype='int64')
init_score = np.ones((bsz, 1), dtype='float32')
parent_idx = np.array(range(bsz), dtype='int64')
tgt_generation_mask = np.ones((bsz, 1, input_seq_len), dtype='float32')
max_dec_len = np.array([max_dec_len], dtype='int64')

src_idx_handle = predictor.get_input_handle('src_ids')
src_idx_handle.reshape([bsz, input_seq_len])
src_idx_handle.copy_from_cpu_bind(src_ids)
print('src_ids shape ', src_ids.shape)

pos_ids_handle = predictor.get_input_handle('pos_ids')
pos_ids_handle.reshape([bsz, input_seq_len])
pos_ids_handle.copy_from_cpu_bind(pos_ids)
print('pos_ids shape ', pos_ids.shape)

input_mask_handle = predictor.get_input_handle('input_mask')
input_mask_handle.reshape([bsz, input_seq_len, input_seq_len])
input_mask_handle.copy_from_cpu_bind(input_mask)
print('input_mask shape ', input_mask.shape)

if generation:
    tgt_ids_handle = predictor.get_input_handle('tgt_ids')
    tgt_ids_handle.reshape([bsz, 1])
    tgt_ids_handle.set_lod(lods)
    print(lods, '================')
    tgt_ids_handle.copy_from_cpu_bind(tgt_ids)

    tgt_pos_handle = predictor.get_input_handle('tgt_pos')
    tgt_pos_handle.reshape([bsz, 1])
#    tgt_pos_handle.set_lod(lods)
    tgt_pos_handle.copy_from_cpu_bind(tgt_pos)

    init_score_handle = predictor.get_input_handle('init_score')
    init_score_handle.reshape([bsz, 1])
    init_score_handle.set_lod(lods)
    init_score_handle.copy_from_cpu_bind(init_score)

    parent_idx_handle = predictor.get_input_handle('parent_idx')
    parent_idx_handle.reshape(parent_idx.shape)
    parent_idx_handle.copy_from_cpu_bind(parent_idx)

    tgt_generation_mask_handle = predictor.get_input_handle('tgt_generation_mask')
    tgt_generation_mask_handle.reshape([bsz, 1, input_seq_len])
    tgt_generation_mask_handle.copy_from_cpu_bind(tgt_generation_mask)

    max_dec_len_handle = predictor.get_input_handle('max_dec_len')
    max_dec_len_handle.reshape([max_dec_len])
    max_dec_len_handle.copy_from_cpu_bind(max_dec_len)

    
output_names = predictor.get_output_names()
max_run_times = 100
start = 0 
skip_times = 20
for i in range(max_run_times):
    if i == skip_times:
        start = time.time()
    if i == max_run_times - 3:
        core.nvprof_start()
    predictor.run()
    for name in output_names:
        handle = predictor.get_output_handle(name)
#        rst = handle.copy_to_cpu()
    if i % 10 == 0:
        print('Finish', i, 'runs')

end = time.time()
print("Time: ", (end - start) / float(max_run_times - skip_times))
