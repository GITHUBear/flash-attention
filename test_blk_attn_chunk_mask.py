from vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
)
import torch
import random

torch.set_default_device("cuda:0")
random.seed(0)
torch.cuda.manual_seed_all(0)

def ceil_div(a, b):
    return (a + b - 1) // b

query_lens = [20, 10]
batch_size = len(query_lens)
kv_lens = [[103, 123, 84, 239, 943, 30],[800,31,435]]
block_size = 16
kv_lens_block_size_align = [[ceil_div(l, block_size)*block_size for l in kv_batch] for kv_batch in kv_lens]
kv_blocks = [[ceil_div(l, block_size) for l in kv_batch] for kv_batch in kv_lens]
print(f"kv_lens_block_size_align:{kv_lens_block_size_align}")
total_kv_len_per_batch = [sum(kv_batch) for kv_batch in kv_lens]
print(f"total_kv_len_per_batch:{total_kv_len_per_batch}")
padded_kv_lens = [align_kv_len[:-1]+kv_len[-1:] for kv_len, align_kv_len in zip(kv_lens, kv_lens_block_size_align)]
print(f"padded_kv_lens:{padded_kv_lens}")
actual_kv_lens_for_blk_attn = [sum(kv_len) for kv_len in padded_kv_lens]
print(f"actual_kv_lens_for_blk_attn: {actual_kv_lens_for_blk_attn}")

num_heads = (5, 1)
head_size = 128
dtype = torch.float16
num_blocks = 8192
fa_version = 2
num_query_heads = num_heads[0]
num_kv_heads = num_heads[1]
window_size = (-1, -1)
scale = head_size**-0.5

query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
max_query_len = max(query_lens)

key = torch.randn(sum(total_kv_len_per_batch),
                    num_kv_heads,
                    head_size,
                    dtype=dtype)
value = torch.rand_like(key)
cu_kv_lens = torch.tensor([0] + total_kv_len_per_batch,
                          dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
max_kv_len = max(total_kv_len_per_batch)


key_cache = torch.randn(num_blocks,
                        block_size,
                        num_kv_heads,
                        head_size,
                        dtype=dtype)
value_cache = torch.randn_like(key_cache)
seqused_k = torch.tensor(actual_kv_lens_for_blk_attn, dtype=torch.int32)
max_kv_len_for_blk_attn = max(actual_kv_lens_for_blk_attn)
max_num_blocks_per_seq_for_blk_attn = (max(actual_kv_lens_for_blk_attn) + block_size - 1) // block_size
num_blk_for_blk_attn = [[(l // block_size) for l in kv_len] for kv_len in kv_lens_block_size_align]
print(f"num_blk_for_blk_attn:{num_blk_for_blk_attn}")
total_blks_per_batch = [sum(num_blks) for num_blks in num_blk_for_blk_attn]
print(f"total_blks_per_batch: {total_blks_per_batch}")
total_blks = sum([num_blk for num_blks in num_blk_for_blk_attn for num_blk in num_blks])
print(f"total_blks: {total_blks}")
blk_ids = random.sample(range(num_blocks), total_blks)
print(f"blk_ids: {blk_ids} max_num_blocks_per_seq_for_blk_attn: {max_num_blocks_per_seq_for_blk_attn}")

block_tables_cpu = []
pre_sum = 0
for i in range(batch_size):
    block_table = [0 for _ in range(max_num_blocks_per_seq_for_blk_attn)]
    block_table[:total_blks_per_batch[i]] = blk_ids[pre_sum:pre_sum+total_blks_per_batch[i]]
    pre_sum += total_blks_per_batch[i]
    block_tables_cpu.append(block_table)
print(f"block_table_cpu: {block_tables_cpu}")
block_tables = torch.tensor(block_tables_cpu, dtype=torch.int32)

pre_kv_sum = 0
for i in range(batch_size):
    pre_blk_cnt_sum = 0
    for (kv_len, num_blk) in zip(kv_lens[i], num_blk_for_blk_attn[i]):
        blocks = block_tables[i][pre_blk_cnt_sum:pre_blk_cnt_sum+num_blk]
        key[pre_kv_sum:pre_kv_sum+kv_len] = (key_cache[blocks].reshape((-1, num_kv_heads, head_size)))[:kv_len]
        value[pre_kv_sum:pre_kv_sum+kv_len] = (value_cache[blocks].reshape((-1, num_kv_heads, head_size)))[:kv_len]

        pre_kv_sum += kv_len
        pre_blk_cnt_sum += num_blk

# [[103, 123, 84, 239, 943, 30]]
# print([l for kv_len in kv_lens for l in kv_len[::-1]])
# print([len(kv_len) for kv_len in kv_lens])
actual_chunked_seqlen_k = torch.tensor([l for kv_len in kv_lens for l in kv_len], dtype=torch.int32)
cu_num_chunks_k = torch.tensor([0] + [len(kv_len) for kv_len in kv_lens],
                          dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
print(cu_num_chunks_k)

output_common = torch.zeros_like(query)
print("===========================")
print(f"cu_query_lens:{cu_query_lens}")
print(f"max_query_len:{max_query_len}")
print(f"cu_kv_lens:{cu_kv_lens}")
print(f"max_kv_len:{max_kv_len}")
print("===========================")
flash_attn_varlen_func(
    q=query,
    k=key,
    v=value,
    cu_seqlens_q=cu_query_lens,
    max_seqlen_q=max_query_len,
    cu_seqlens_k=cu_kv_lens,
    max_seqlen_k=max_kv_len,
    softmax_scale=scale,
    causal=True,
    out=output_common,
    fa_version=fa_version,
)
torch.cuda.synchronize()
# print(output_common)

output_chunk = torch.zeros_like(query)
print("===========================")
print(f"cu_query_lens:{cu_query_lens}")
print(f"max_query_len:{max_query_len}")
print(f"seqused_k:{seqused_k}")
print(f"max_kv_len:{max_kv_len_for_blk_attn}")
print(f"block_tables:{block_tables}")
print(f"actual_chunked_seqlen_k:{actual_chunked_seqlen_k}")
print(f"cu_num_chunks_k:{cu_num_chunks_k}")
print("===========================")
flash_attn_varlen_func(
    q=query,
    k=key_cache,
    v=value_cache,
    cu_seqlens_q=cu_query_lens,
    max_seqlen_q=max_query_len,
    seqused_k=seqused_k,
    max_seqlen_k=max_kv_len_for_blk_attn,
    softmax_scale=scale,
    causal=True,
    block_table=block_tables,
    actual_chunked_seqlen_k=actual_chunked_seqlen_k,
    cu_num_chunks_k=cu_num_chunks_k,
    out=output_chunk,
    fa_version=fa_version,
)
torch.cuda.synchronize()
# print(output_chunk)

print(torch.abs(output_common - output_chunk).max())
# print(torch.allclose(output, output_tmp, atol=1e-3))

# print("XXXXXXXXXXXXXXXXXXXXXXXX")
# print(torch.abs(key[:3] - key_cache[6311, :3]).max())
# print(torch.abs(key[3:] - key_cache[6890, :3]).max())
# print(torch.abs(value[:3] - value_cache[6311, :3]).max())
# print(torch.abs(value[3:] - value_cache[6890, :3]).max())