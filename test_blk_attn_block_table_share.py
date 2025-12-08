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

block_size = 16
kv_lens = [[10, 12, 40, 239, 10000, 20], [10, 1000]]
# kv_lens = [[10, 12, 40, 239, 10000, 20]]
# kv_lens = [[1,1],[1]]
print(f"kv_lens: {kv_lens}")
rotray_offsets = [[0,0,0,0,0,0], [0, 0]]
# rotray_offsets = [[0,0,0,0,0,0]]
# rotray_offsets = [[0,0], [0]]
batch_size = len(kv_lens)

kv_lens_block_size_align = [[ceil_div(l, block_size)*block_size for l in kv_batch] for kv_batch in kv_lens]
kv_blocks = [[ceil_div(l, block_size) for l in kv_batch] for kv_batch in kv_lens]

padded_kv_lens = [align_kv_len[:-1]+kv_len[-1:] for kv_len, align_kv_len in zip(kv_lens, kv_lens_block_size_align)]
print(f"padded_kv_lens:{padded_kv_lens}")
total_kv_len_per_batch = [sum(kv_batch) for kv_batch in kv_lens]
print(f"total_kv_len_per_batch:{total_kv_len_per_batch}")
padded_kv_len_per_batch = [sum(padded_kv_len) for padded_kv_len in padded_kv_lens]
print(f"padded_kv_len_per_batch:{padded_kv_len_per_batch}")

query_lens = [kv_len for kv_lens in padded_kv_lens for kv_len in kv_lens]
print(f"query_lens: {query_lens}")
seqlen_kv = []
for padded_kv_len in padded_kv_lens:
    seqlen_kv.append(padded_kv_len[:-1] + [sum(padded_kv_len)])
print(f"seqlen_kv: {seqlen_kv}")
seqlen_kv_origin = []
for padded_kv_len, origin_kv_lens in zip(padded_kv_lens, kv_lens):
    seqlen_kv_origin.append(padded_kv_len[:-1] + [sum(origin_kv_lens)])
print(f"seqlen_kv_origin: {seqlen_kv_origin}")
flattened_seqlen_kv = [seqlen for seqlens in seqlen_kv for seqlen in seqlens]
print(f"flattened_seqlen_kv: {flattened_seqlen_kv}")
flattened_seqlen_kv_origin = [seqlen for seqlens in seqlen_kv_origin for seqlen in seqlens]
print(f"flattened_seqlen_kv_origin: {flattened_seqlen_kv_origin}")

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
key_cache = torch.randn(num_blocks,
                        block_size,
                        num_kv_heads,
                        head_size,
                        dtype=dtype)
value_cache = torch.randn_like(key_cache)
seqused_k = torch.tensor(flattened_seqlen_kv, dtype=torch.int32)
key = torch.randn(sum(flattened_seqlen_kv_origin),
                    num_kv_heads,
                    head_size,
                    dtype=dtype)
value = torch.rand_like(key)
cu_kv_lens = torch.tensor([0] + flattened_seqlen_kv,
                          dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
cu_kv_lens_origin = torch.tensor([0] + flattened_seqlen_kv_origin,
                          dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
max_kv_len = max(flattened_seqlen_kv)
max_kv_len_origin = max(flattened_seqlen_kv_origin)

total_blks_per_batch = [ceil_div(seqlen[-1], block_size) for seqlen in seqlen_kv]
total_blks = sum(total_blks_per_batch)
blk_ids = random.sample(range(num_blocks), total_blks)

block_tables_cpu = []
max_num_block_per_batch = 2000
assert max_num_block_per_batch > max(total_blks_per_batch)
pre_sum = 0
for i in range(batch_size):
    block_table = [0 for _ in range(max_num_block_per_batch)]
    block_table[:total_blks_per_batch[i]] = blk_ids[pre_sum:pre_sum+total_blks_per_batch[i]]
    pre_sum += total_blks_per_batch[i]
    block_tables_cpu.append(block_table)
block_tables = torch.tensor(block_tables_cpu, dtype=torch.int32)
print(f"block_tables shape: {block_tables.shape}")


# cos_sin_cache
base = 1000000.0
rotary_dim = 128
max_position_embeddings = 4096
inv_freq = 1.0 / (base**(torch.arange(
            0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
t = torch.arange(max_position_embeddings, dtype=torch.float)
freqs = torch.einsum("i,j -> ij", t, inv_freq)
cos = freqs.cos()
sin = freqs.sin()
cos_sin_cache = torch.cat((cos, sin), dim=-1).to(dtype=query.dtype)
assert cos_sin_cache.stride(-1) == 1
##############

def apply_rotary_emb_torch(
    x: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    is_neg = (offset < 0)
    if is_neg:
        offset = -offset
    cos_sin = cos_sin_cache[offset]
    # print(f"offset: {cos_sin}")
    cos, sin = cos_sin.chunk(2, dim=-1)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin * (-1 if is_neg else 1)
    o2 = x2 * cos + x1 * sin * (-1 if is_neg else 1)
    return torch.cat((o1, o2), dim=-1)


# 为 key、value 赋值
block_table_offsets_cpu = []
cur_kv_offset = 0
for bidx in range(batch_size):
    global_offset = bidx * max_num_block_per_batch
    cur_block_table_offsets = []

    seqlens = seqlen_kv_origin[bidx]
    actual_seqlens = kv_lens[bidx]
    cur_blk_table = block_tables[bidx]
    cur_rotary_offset = rotray_offsets[bidx]
    cur_blk_offset = 0
    last_kv_offset = cur_kv_offset + sum(seqlens[:-1])
    for seqlen, roffset, actual_seqlen in zip(seqlens[:-1], cur_rotary_offset[:-1], actual_seqlens[:-1]):
        num_blk = ceil_div(seqlen, block_size)
        blocks = cur_blk_table[cur_blk_offset:cur_blk_offset+num_blk]
        cur_block_table_offsets.append(global_offset + cur_blk_offset)

        print(f"load to {cur_kv_offset} -> {cur_kv_offset + seqlen} & {last_kv_offset} -> {last_kv_offset + actual_seqlen}")
        if roffset != 0:
            key_before_rotray = (key_cache[blocks].reshape((-1, num_kv_heads, head_size)))[:seqlen].clone()
            key[cur_kv_offset:cur_kv_offset+seqlen] = apply_rotary_emb_torch(key_before_rotray, roffset)
        else:
            key[cur_kv_offset:cur_kv_offset+seqlen] = (key_cache[blocks].reshape((-1, num_kv_heads, head_size)))[:seqlen]
        key[last_kv_offset:last_kv_offset+actual_seqlen] = key[cur_kv_offset:cur_kv_offset+actual_seqlen]
        value[cur_kv_offset:cur_kv_offset+seqlen] = (value_cache[blocks].reshape((-1, num_kv_heads, head_size)))[:seqlen]
        value[last_kv_offset:last_kv_offset+actual_seqlen] = value[cur_kv_offset:cur_kv_offset+actual_seqlen]

        cur_blk_offset += num_blk
        cur_kv_offset += seqlen
        last_kv_offset += actual_seqlen
    
    cur_block_table_offsets.append(global_offset)
    block_table_offsets_cpu.extend(cur_block_table_offsets)
    roffset = cur_rotary_offset[-1]
    seqlen = actual_seqlens[-1]
    num_blk = ceil_div(seqlen, block_size)
    blocks = cur_blk_table[cur_blk_offset:cur_blk_offset+num_blk]
    assert roffset == 0
    print(f"last load to {last_kv_offset} -> {last_kv_offset + seqlen}")
    key[last_kv_offset:last_kv_offset+seqlen] = (key_cache[blocks].reshape((-1, num_kv_heads, head_size)))[:seqlen]
    value[last_kv_offset:last_kv_offset+seqlen] = (value_cache[blocks].reshape((-1, num_kv_heads, head_size)))[:seqlen]
    cur_kv_offset = last_kv_offset+seqlen
print(f"block_table_offsets_cpu: {block_table_offsets_cpu}")
block_table_offsets_gpu = torch.tensor(block_table_offsets_cpu, dtype=torch.int32)


actual_chunked_seqlen_cpu = []
rotary_cpu = []
num_chunk = []
for bidx in range(batch_size):
    cur_rotary = rotray_offsets[bidx]
    cur_seqlen = seqlen_kv[bidx]
    cur_actual_kv_len = kv_lens[bidx]

    chunks = len(cur_rotary)
    num_chunk.extend([1 for _ in range(chunks - 1)])
    num_chunk.append(chunks)

    actual_chunked_seqlen_cpu.extend(cur_seqlen[:-1])
    actual_chunked_seqlen_cpu.extend(cur_actual_kv_len)

    rotary_cpu.extend(cur_rotary[:-1])
    rotary_cpu.extend(cur_rotary)
print(f"actual_chunked_seqlen_cpu: {actual_chunked_seqlen_cpu}")
print(f"rotary_cpu: {rotary_cpu}")
print(f"num_chunk: {num_chunk}")
actual_chunked_seqlen_tensor = torch.tensor(actual_chunked_seqlen_cpu, dtype=torch.int32)
rotary_tensor = torch.tensor(rotary_cpu, dtype=torch.int32)
cu_num_chunks_k = torch.tensor([0] + num_chunk,
                          dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)

output_common = torch.zeros_like(query)
print("===========================")
print(f"qshape: {query.shape}")
print(f"kshape: {key.shape}")
print(f"vshape: {value.shape}")
print(f"cu_query_lens:{cu_query_lens}")
print(f"max_query_len:{max_query_len}")
print(f"cu_kv_lens_origin:{cu_kv_lens_origin}")
print(f"max_kv_len_origin:{max_kv_len_origin}")
print("===========================")
flash_attn_varlen_func(
    q=query,
    k=key,
    v=value,
    cu_seqlens_q=cu_query_lens,
    max_seqlen_q=max_query_len,
    cu_seqlens_k=cu_kv_lens_origin,
    max_seqlen_k=max_kv_len_origin,
    softmax_scale=scale,
    causal=True,
    out=output_common,
    fa_version=fa_version,
)
torch.cuda.synchronize()

print(f"seqused_k: {seqused_k}")
print(f"block_table: {block_tables}")
# print(f"key1: {key_cache[6311].reshape((-1, num_kv_heads, head_size))}")
# print(f"key2: {key_cache[6890].reshape((-1, num_kv_heads, head_size))}")
# print(f"key3: {key_cache[663].reshape((-1, num_kv_heads, head_size))}")
# print(f"key: {key}")
output_chunk = torch.zeros_like(query)
flash_attn_varlen_func(
    q=query,
    k=key_cache,
    v=value_cache,
    cu_seqlens_q=cu_query_lens,
    max_seqlen_q=max_query_len,
    seqused_k=seqused_k,
    max_seqlen_k=max_kv_len,
    softmax_scale=scale,
    causal=True,
    block_table=block_tables,
    
    actual_chunked_seqlen_k=actual_chunked_seqlen_tensor,
    chunk_rotray_offset_positions=rotary_tensor,
    cu_num_chunks_k=cu_num_chunks_k,
    cos_sin_cache=cos_sin_cache,
    enable_splitkv_for_chunked_kv=True,
    block_table_offsets=block_table_offsets_gpu,

    out=output_chunk,
    fa_version=fa_version,
)
torch.cuda.synchronize()

print(torch.abs(output_common - output_chunk).max())
# print(f"output_shape: {output_common.shape}")
# print(f"output_common: {output_common}")
# print(f"output_chunk: {output_chunk}")