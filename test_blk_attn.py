from vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
    is_fa_version_supported,
)
import torch
import random

torch.set_default_device("cuda:0")
random.seed(0)
torch.cuda.manual_seed_all(0)

# seq_lens = [[20, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 20] * 10]
# seq_lens = [[20, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 20] * 10]
# seq_lens = [[20, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 20] * 10]
doc_len = 1000
num_doc = 10
batch_size = 10
seq_lens = [[20] + [doc_len] * num_doc + [20]] * batch_size
print(seq_lens)
num_heads = (5, 1)
head_size = 128
dtype = torch.float16
num_blocks = 8192
block_size = 16
fa_version = 2

num_seqs = len(seq_lens)
query_lens = [sum(x) for x in seq_lens]
kv_lens = [sum(x) for x in seq_lens]
num_query_heads = num_heads[0]
num_kv_heads = num_heads[1]

window_size = (-1, -1)
scale = head_size**-0.5

query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
key_cache = torch.randn(num_blocks,
                        block_size,
                        num_kv_heads,
                        head_size,
                        dtype=dtype)
value_cache = torch.randn_like(key_cache)
kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
query_lens_tensor = torch.tensor(query_lens, dtype=torch.int32)
max_seqlen_q = max(query_lens)
cu_query_lens = torch.tensor([0] + query_lens,
                             dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
seqused_k = torch.tensor(kv_lens, dtype=torch.int32)
max_seqlen_k = max(kv_lens)
max_num_blocks_per_seq = (max_seqlen_k + block_size - 1) // block_size
block_tables = torch.randint(0, num_blocks,
                            (num_seqs, max_num_blocks_per_seq),
                            dtype=torch.int32)

flatten_query_lens = [l for g in seq_lens for l in g]
flatten_kv_lens = []
for g in seq_lens:
    sum_l = 0
    for idx, l in enumerate(g):
        sum_l += l
        if idx == len(g) - 1:
            flatten_kv_lens.append(sum_l)
        else:
            flatten_kv_lens.append(l)
flatten_cu_query_lens = torch.tensor([0] + flatten_query_lens,
                             dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
flatten_max_seqlen_q = max(flatten_query_lens)
flatten_seqused_k = torch.tensor(flatten_kv_lens, dtype=torch.int32)
flatten_max_seqlen_k = max(flatten_kv_lens)
flatten_max_num_blocks_per_seq = (flatten_max_seqlen_k + block_size - 1) // block_size
flatten_block_tables = torch.randint(0, num_blocks,
                            (len(flatten_query_lens), flatten_max_num_blocks_per_seq),
                            dtype=torch.int32)

ph_query_lens = [g[-1] for g in seq_lens]
ph_kv_lens = [sum(g) for g in seq_lens]
ph_cu_query_lens = torch.tensor([0] + ph_query_lens,
                             dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
ph_max_seqlen_q = max(ph_query_lens)
ph_seqused_k = torch.tensor(ph_kv_lens, dtype=torch.int32)
ph_max_seqlen_k = max(ph_kv_lens)
ph_max_num_blocks_per_seq = (ph_max_seqlen_k + block_size - 1) // block_size
ph_block_tables = torch.randint(0, num_blocks,
                            (len(ph_query_lens), ph_max_num_blocks_per_seq),
                            dtype=torch.int32)
ph_query = torch.randn(sum(ph_query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)


# print(f"flatten_query_lens: {flatten_query_lens}")
# print(f"flatten_kv_lens: {flatten_kv_lens}")
print(f"flatten_block table shape: {flatten_block_tables.shape}")
# print(f"flatten_cu_query_lens: {flatten_cu_query_lens}")
print(f"flatten_max_seqlen_q: {flatten_max_seqlen_q}")
# print(f"flatten_seqused_k: {flatten_seqused_k}")
print(f"flatten_max_seqlen_k: {flatten_max_seqlen_k}")
print("=======================")

# print("block table:")
# print(block_tables)
print(f"block table shape: {block_tables.shape}")
print(f"query.shape: {query.shape}")
print(f"key_cache.shape: {key_cache.shape}")
print(f"value_cache.shape: {value_cache.shape}")
print(f"cu_seqlen_q: {cu_query_lens}")
print(f"max_seqlen_q: {max_seqlen_q}")
print(f"seqused_k: {seqused_k}")
print(f"max_seqlen_k: {max_seqlen_k}")

repeat_times = 10
start_event1 = torch.cuda.Event(enable_timing=True)
end_event1 = torch.cuda.Event(enable_timing=True)
output_tmp = torch.zeros_like(query)
for _ in range(repeat_times):
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        block_table=block_tables,
        out=output_tmp,
        fa_version=fa_version,
    )
torch.cuda.synchronize()

output = torch.zeros_like(query)
start_event1.record()
for _ in range(repeat_times):
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        block_table=block_tables,
        out=output,
        fa_version=fa_version,
    )
end_event1.record()
torch.cuda.synchronize()
elapsed_time_ms1 = start_event1.elapsed_time(end_event1)
print(f"cuda cost: {elapsed_time_ms1/repeat_times}ms")


start_event2 = torch.cuda.Event(enable_timing=True)
end_event2 = torch.cuda.Event(enable_timing=True)
output_tmp = torch.zeros_like(query)
for _ in range(repeat_times):
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=flatten_cu_query_lens,
        max_seqlen_q=flatten_max_seqlen_q,
        seqused_k=flatten_seqused_k,
        max_seqlen_k=flatten_max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        block_table=flatten_block_tables,
        out=output_tmp,
        fa_version=fa_version,
    )
torch.cuda.synchronize()

output = torch.zeros_like(query)
start_event2.record()
for _ in range(repeat_times):
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=flatten_cu_query_lens,
        max_seqlen_q=flatten_max_seqlen_q,
        seqused_k=flatten_seqused_k,
        max_seqlen_k=flatten_max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        block_table=flatten_block_tables,
        out=output,
        fa_version=fa_version,
    )
end_event2.record()
torch.cuda.synchronize()
elapsed_time_ms2 = start_event2.elapsed_time(end_event2)
print(f"blk cuda cost: {elapsed_time_ms2/repeat_times}ms")



start_event3 = torch.cuda.Event(enable_timing=True)
end_event3 = torch.cuda.Event(enable_timing=True)
output_tmp = torch.zeros_like(ph_query)
for _ in range(repeat_times):
    flash_attn_varlen_func(
        q=ph_query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=ph_cu_query_lens,
        max_seqlen_q=ph_max_seqlen_q,
        seqused_k=ph_seqused_k,
        max_seqlen_k=ph_max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        block_table=ph_block_tables,
        out=output_tmp,
        fa_version=fa_version,
    )
torch.cuda.synchronize()

output = torch.zeros_like(ph_query)
start_event3.record()
for _ in range(repeat_times):
    flash_attn_varlen_func(
        q=ph_query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=ph_cu_query_lens,
        max_seqlen_q=ph_max_seqlen_q,
        seqused_k=ph_seqused_k,
        max_seqlen_k=ph_max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        block_table=ph_block_tables,
        out=output,
        fa_version=fa_version,
    )
end_event3.record()
torch.cuda.synchronize()
elapsed_time_ms3 = start_event3.elapsed_time(end_event3)
print(f"ph cuda cost: {elapsed_time_ms3/repeat_times}ms")
print(output[0])