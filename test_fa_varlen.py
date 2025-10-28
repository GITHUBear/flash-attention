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

# seq_lens = [(1, 5000), (32, 13234), (1, 1234), (1, 32000), (4, 32), (5, 4500), (1, 65536)]
seq_lens = [(1, 30000)] * 10
num_heads = (5, 1)
head_size = 128
dtype = torch.float16
num_blocks = 8192
block_size = 16
fa_version = 2

num_page_compress_cache = 100
page_compress_topk = 256

num_seqs = len(seq_lens)
query_lens = [x[0] for x in seq_lens]
kv_lens = [x[1] for x in seq_lens]
num_query_heads = num_heads[0]
num_kv_heads = num_heads[1]
assert num_query_heads % num_kv_heads == 0

max_query_len = max(query_lens)
max_kv_len = max(kv_lens)
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
max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
block_tables = torch.randint(0, num_blocks,
                            (num_seqs, max_num_blocks_per_seq),
                            dtype=torch.int32)
print("block table:")
print(block_tables)
print("block table shape:")
print(block_tables.shape)

page_compress_cache = torch.randint(0, num_blocks, 
                                    (num_page_compress_cache, num_kv_heads, page_compress_topk),
                                    dtype=torch.int32)
print(f"page_compress_cache.shape: {page_compress_cache.shape}")

page_compress_cache_ids_mask = (kv_lens_tensor < page_compress_topk * block_size)
page_compress_cache_ids_mask2 = (query_lens_tensor > 1)
page_compress_cache_ids = torch.randint(0, num_page_compress_cache, (num_seqs,), dtype=torch.int32)
page_compress_cache_ids[page_compress_cache_ids_mask] = -1
page_compress_cache_ids[page_compress_cache_ids_mask2] = -1
page_compress_cache_ids_list = page_compress_cache_ids.tolist()
print("page_compress_cache_ids: ")
print(page_compress_cache_ids)
print("page_compress_cache_ids shape: ")
print(page_compress_cache_ids.shape)

num_compressed_pages = []
for kv_len in kv_lens:
    if page_compress_topk > kv_len // block_size:
        num_compressed_pages.append(-1)
    else:
        # num_compressed_pages.append(random.randint(page_compress_topk, kv_len // block_size))
        num_compressed_pages.append(kv_len // block_size)
num_compressed_pages = torch.tensor(num_compressed_pages, dtype=torch.int32)
num_compressed_pages_list = num_compressed_pages.tolist()
print("num_compressed_pages: ")
print(num_compressed_pages)
print("num_compressed_pages shape: ")
print(num_compressed_pages.shape)

new_kv_lens = []
for cache_id, n_cprs, kv_len in zip(page_compress_cache_ids_list, num_compressed_pages_list, kv_lens):
    if cache_id == -1:
        new_kv_lens.append(kv_len)
    else:
        new_kv_lens.append(kv_len - n_cprs * block_size + page_compress_topk * block_size)
actual_max_kv_len = max(new_kv_lens)
new_kv_lens = torch.tensor(new_kv_lens, dtype=torch.int32)
new_kv_lens_list = new_kv_lens.tolist()
print("new_kv_lens: ")
print(new_kv_lens)
print("new_kv_lens shape: ")
print(new_kv_lens.shape)
print()

repeat_times = 10
start_event1 = torch.cuda.Event(enable_timing=True)
end_event1 = torch.cuda.Event(enable_timing=True)
start_event2 = torch.cuda.Event(enable_timing=True)
end_event2 = torch.cuda.Event(enable_timing=True)

std_output = torch.zeros_like(query)
group_size = num_query_heads // num_kv_heads
start_event1.record()
for _ in range(repeat_times):
    for bid in range(num_seqs):
        for qhead_i in range(num_query_heads):
            kvhead_i = qhead_i // group_size
            page_compress_cache_id = page_compress_cache_ids[bid].item()
            qlen = query_lens[bid]
            qoffset = cu_query_lens[bid].item()
            single_query_cu_seqlen = torch.tensor([0, qlen], dtype=torch.int32)
            if page_compress_cache_id == -1:
                flash_attn_varlen_func(
                    q=query[qoffset:qoffset+qlen, qhead_i:qhead_i+1, :],
                    k=key_cache[:, :, kvhead_i:kvhead_i+1, :],
                    v=value_cache[:, :, kvhead_i:kvhead_i+1, :],
                    cu_seqlens_q=single_query_cu_seqlen,
                    max_seqlen_q=qlen,
                    seqused_k=new_kv_lens[bid:bid+1],
                    max_seqlen_k=new_kv_lens_list[bid],
                    softmax_scale=scale,
                    causal=True,
                    block_table=block_tables[bid:bid+1, :],
                    out=std_output[qoffset:qoffset+qlen, qhead_i:qhead_i+1, :],
                    fa_version=fa_version,
                )
            else:
                assert qlen == 1
                num_cprs_page = num_compressed_pages[bid:bid+1].item()
                compressed_page_ids = page_compress_cache[page_compress_cache_id, kvhead_i, :]
                uncompressed_page_ids = block_tables[bid, num_cprs_page:]
                new_block_table = torch.concat([compressed_page_ids, uncompressed_page_ids]).unsqueeze(0)
                flash_attn_varlen_func(
                    q=query[qoffset:qoffset+1, qhead_i:qhead_i+1, :],
                    k=key_cache[:, :, kvhead_i:kvhead_i+1, :],
                    v=value_cache[:, :, kvhead_i:kvhead_i+1, :],
                    cu_seqlens_q=single_query_cu_seqlen,
                    max_seqlen_q=1,
                    seqused_k=new_kv_lens[bid:bid+1],
                    max_seqlen_k=new_kv_lens_list[bid],
                    softmax_scale=scale,
                    causal=True,
                    block_table=new_block_table,
                    out=std_output[qoffset:qoffset+1, qhead_i:qhead_i+1, :],
                    fa_version=fa_version,
                )    
end_event1.record()
torch.cuda.synchronize()
print("std_output: ")
print(std_output)
print("std_output shape: ")
print(std_output.shape)
elapsed_time_ms = start_event1.elapsed_time(end_event1)
print(f"torch cost: {elapsed_time_ms/10}ms")

output_tmp = torch.zeros_like(query)
for _ in range(repeat_times):
    flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=max_seqlen_q,
        seqused_k=new_kv_lens,
        max_seqlen_k=actual_max_kv_len,
        softmax_scale=scale,
        causal=True,
        block_table=block_tables,
        page_compress_cache=page_compress_cache,
        page_compress_cache_ids=page_compress_cache_ids,
        num_compressed_pages=num_compressed_pages,
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
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=max_seqlen_q,
        seqused_k=new_kv_lens,
        max_seqlen_k=actual_max_kv_len,
        softmax_scale=scale,
        causal=True,
        block_table=block_tables,
        page_compress_cache=page_compress_cache,
        page_compress_cache_ids=page_compress_cache_ids,
        num_compressed_pages=num_compressed_pages,
        out=output,
        fa_version=fa_version,
    )
end_event2.record()
torch.cuda.synchronize()
print("output: ")
print(output)
print("output shape: ")
print(output.shape)
elapsed_time_ms2 = start_event2.elapsed_time(end_event2)
print(f"cuda cost: {elapsed_time_ms2/repeat_times}ms")

start_event3 = torch.cuda.Event(enable_timing=True)
end_event3 = torch.cuda.Event(enable_timing=True)
start_event3.record()
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
end_event3.record()
torch.cuda.synchronize()
elapsed_time_ms3 = start_event3.elapsed_time(end_event3)
print(f"origin cost: {elapsed_time_ms3/repeat_times}ms")

print(torch.abs(output - std_output).max())
print(torch.allclose(output, std_output, atol=3e-4))