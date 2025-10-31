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

doc_len = 1000
num_doc = 10
batch_size = 10
seq_lens = [[20] + [doc_len] * num_doc + [20]] * batch_size
print(seq_lens)
batch_idx_offset_for_blk_attn = []
for g in seq_lens:
    batch_idx_offset_for_blk_attn.extend([0] * (len(g) - 1) + [len(g) - 1])
print(batch_idx_offset_for_blk_attn)

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
key = torch.randn(sum(kv_lens),
                        num_kv_heads,
                        head_size,
                        dtype=dtype)
value = torch.randn_like(key)
# kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
# query_lens_tensor = torch.tensor(query_lens, dtype=torch.int32)
max_seqlen_q = max(query_lens)
max_seqlen_k = max(kv_lens)
cu_query_lens = torch.tensor([0] + query_lens,
                             dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
cu_key_lens = torch.tensor([0] + kv_lens,
                           dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)

flatten_query_lens = [l for g in seq_lens for l in g]
flatten_kv_lens = [l for g in seq_lens for l in g]
flatten_cu_query_lens = torch.tensor([0] + flatten_query_lens,
                             dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
flatten_max_seqlen_q = max(flatten_query_lens)
flatten_cu_kv_lens = torch.tensor([0] + flatten_kv_lens,
                             dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
batch_idx_offset_for_blk_attn_tensor = torch.tensor(batch_idx_offset_for_blk_attn, dtype=torch.int32)

repeat_times = 10
start_event1 = torch.cuda.Event(enable_timing=True)
end_event1 = torch.cuda.Event(enable_timing=True)
output_tmp = torch.zeros_like(query)
for _ in range(repeat_times):
    flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_k=cu_key_lens,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        out=output_tmp,
        fa_version=fa_version,
    )
torch.cuda.synchronize()

output_tmp = torch.zeros_like(query)
start_event1.record()
for _ in range(repeat_times):
    flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_k=cu_key_lens,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        out=output_tmp,
        fa_version=fa_version,
    )
end_event1.record()
torch.cuda.synchronize()
elapsed_time_ms1 = start_event1.elapsed_time(end_event1)
print(f"cuda cost: {elapsed_time_ms1/repeat_times}ms")
# print(output_tmp[0])


start_event2 = torch.cuda.Event(enable_timing=True)
end_event2 = torch.cuda.Event(enable_timing=True)
output_tmp = torch.zeros_like(query)
for _ in range(repeat_times):
    flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=flatten_cu_query_lens,
        max_seqlen_q=flatten_max_seqlen_q,
        cu_seqlens_k=flatten_cu_kv_lens,
        batch_idx_offset_for_blk_attn=batch_idx_offset_for_blk_attn_tensor,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        out=output_tmp,
        fa_version=fa_version,
    )
torch.cuda.synchronize()

output = torch.zeros_like(query)
start_event2.record()
for _ in range(repeat_times):
    flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=flatten_cu_query_lens,
        max_seqlen_q=flatten_max_seqlen_q,
        cu_seqlens_k=flatten_cu_kv_lens,
        batch_idx_offset_for_blk_attn=batch_idx_offset_for_blk_attn_tensor,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        out=output,
        fa_version=fa_version,
    )
end_event2.record()
torch.cuda.synchronize()
elapsed_time_ms2 = start_event2.elapsed_time(end_event2)
print(f"cuda cost: {elapsed_time_ms2/repeat_times}ms")


output2 = torch.zeros_like(query)
batch_start = 0
blk_start = 0
for g in seq_lens:
    cur_batch_len = 0
    for idx, l in enumerate(g):
        blk_end = blk_start + l
        if idx == len(g) - 1:
            flash_attn_varlen_func(
                q=query[blk_start : blk_end],
                k=key[batch_start : blk_end],
                v=value[batch_start : blk_end],
                cu_seqlens_q=torch.tensor([0, l], dtype=torch.int32),
                max_seqlen_q=l,
                cu_seqlens_k=torch.tensor([0, blk_end - batch_start], dtype=torch.int32),
                max_seqlen_k=(blk_end - batch_start),
                softmax_scale=scale,
                causal=True,
                out=output2[blk_start : blk_end],
                fa_version=fa_version,
            )
            torch.cuda.synchronize()
        else:
            flash_attn_varlen_func(
                q=query[blk_start : blk_end],
                k=key[blk_start : blk_end],
                v=value[blk_start : blk_end],
                cu_seqlens_q=torch.tensor([0, l], dtype=torch.int32),
                max_seqlen_q=l,
                cu_seqlens_k=torch.tensor([0, l], dtype=torch.int32),
                max_seqlen_k=l,
                softmax_scale=scale,
                causal=True,
                out=output2[blk_start : blk_end],
                fa_version=fa_version,
            )
            torch.cuda.synchronize()
        blk_start += l
        cur_batch_len += l
    batch_start += cur_batch_len

# print(output2[0])
# print("=====================")
# print(output[0])
# print(query[0])

print(torch.abs(output - output2).max())
print(torch.allclose(output, output2, atol=3e-4))
# print(query)