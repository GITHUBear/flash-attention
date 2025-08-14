from vllm_flash_attn.flash_attn_interface import flash_attn_with_kvcache
import random
from typing import List, Optional, Tuple

import pytest
import torch

fa_version = 2
window_size = (-1, -1)
num_page_compress_cache = 100
soft_cap = None

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16]
DTYPES = [torch.float16, torch.bfloat16]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]


def ref_page_select_out(query, num_query_heads, num_kv_heads, num_seqs, page_compress_cache_ids, 
                        block_tables, key_cache, value_cache, scale, new_kv_lens, num_compressed_pages,
                        page_compress_cache):
    std_output = torch.zeros_like(query).unsqueeze(1)
    group_size = num_query_heads // num_kv_heads
    for bid in range(num_seqs):
        for qhead_i in range(num_query_heads):
            kvhead_i = qhead_i // group_size
            # 计算 block_table
            page_compress_cache_id = page_compress_cache_ids[bid].item()
            if page_compress_cache_id == -1:
                block_table = block_tables[bid:bid+1, :]
                flash_attn_with_kvcache(
                    query[bid:bid+1, qhead_i:qhead_i+1, :].unsqueeze(1),
                    key_cache[:, :, kvhead_i:kvhead_i+1, :],
                    value_cache[:, :, kvhead_i:kvhead_i+1, :],
                    softmax_scale=scale,
                    causal=True,
                    block_table=block_table,
                    cache_seqlens=new_kv_lens[bid:bid+1],
                    fa_version=fa_version,
                    out=std_output[bid:bid+1, :, qhead_i:qhead_i+1, :]
                )
            else:
                num_cprs_page = num_compressed_pages[bid:bid+1].item()
                compressed_page_ids = page_compress_cache[page_compress_cache_id, kvhead_i, :]
                uncompressed_page_ids = block_tables[bid, num_cprs_page:]
                new_block_table = torch.concat([compressed_page_ids, uncompressed_page_ids]).unsqueeze(0)
                flash_attn_with_kvcache(
                    query[bid:bid+1, qhead_i:qhead_i+1, :].unsqueeze(1),
                    key_cache[:, :, kvhead_i:kvhead_i+1, :],
                    value_cache[:, :, kvhead_i:kvhead_i+1, :],
                    softmax_scale=scale,
                    causal=True,
                    block_table=new_block_table,
                    cache_seqlens=new_kv_lens[bid:bid+1],
                    fa_version=fa_version,
                    out=std_output[bid:bid+1, :, qhead_i:qhead_i+1, :]
                )
    return std_output

@pytest.mark.parametrize("kv_lens", [[4096, 8192, 10240]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("page_compress_topk", [64, 128, 256])
@pytest.mark.parametrize("non_compress_p", [0.4, 0.5, 0.3])
@torch.inference_mode()
def test_flash_attn_with_paged_kv_with_kv_cache(
        kv_lens: List[int],
        num_heads: Tuple[int, int],
        head_size: int,
        dtype: torch.dtype,
        block_size: int,
        soft_cap: Optional[float],
        num_blocks: int,
        page_compress_topk: int,
        non_compress_p: float,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(num_blocks,
                        block_size,
                        num_kv_heads,
                        head_size,
                        dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    page_compress_cache = torch.randint(0, num_blocks, 
                                        (num_page_compress_cache, num_kv_heads, page_compress_topk),
                                        dtype=torch.int32)
    page_compress_cache_ids_mask = torch.rand(num_seqs) < non_compress_p
    page_compress_cache_ids = torch.randint(0, num_page_compress_cache, (num_seqs,), dtype=torch.int32)
    page_compress_cache_ids[page_compress_cache_ids_mask] = -1
    page_compress_cache_ids_list = page_compress_cache_ids.tolist()
    num_compressed_pages = []
    for kv_len in kv_lens:
        num_compressed_pages.append(random.randint(page_compress_topk, kv_len // block_size))
    num_compressed_pages = torch.tensor(num_compressed_pages, dtype=torch.int32)
    num_compressed_pages_list = num_compressed_pages.tolist()
    new_kv_lens = []
    for cache_id, n_cprs, kv_len in zip(page_compress_cache_ids_list, num_compressed_pages_list, kv_lens):
        if cache_id == -1:
            new_kv_lens.append(kv_len)
        else:
            new_kv_lens.append(kv_len - n_cprs * block_size + page_compress_topk * block_size)
    actual_max_kv_len = max(new_kv_lens)
    actual_max_num_blocks_per_seq = (actual_max_kv_len + block_size - 1) // block_size
    new_kv_lens = torch.tensor(new_kv_lens, dtype=torch.int32)

    output = flash_attn_with_kvcache(
        q = query.unsqueeze(1),
        k_cache= key_cache,
        v_cache= value_cache,
        block_table= block_tables,
        cache_seqlens= new_kv_lens,
        softmax_scale= scale,
        causal= True,
        window_size= window_size,
        softcap= soft_cap if soft_cap is not None else 0,
        fa_version= fa_version,
        page_compress_cache=page_compress_cache,
        page_compress_cache_ids=page_compress_cache_ids,
        num_compressed_pages=num_compressed_pages,
        actual_max_num_blocks_per_seq=actual_max_num_blocks_per_seq,
    ).squeeze(1)
    torch.cuda.synchronize()
    ref_output = ref_page_select_out(
        query, 
        num_query_heads, 
        num_kv_heads, 
        num_seqs, 
        page_compress_cache_ids, 
        block_tables, 
        key_cache, 
        value_cache, 
        scale, 
        new_kv_lens, 
        num_compressed_pages,
        page_compress_cache
    ).squeeze(1)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=2e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"