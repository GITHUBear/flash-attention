from vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
    is_fa_version_supported,
)
import torch
import random

torch.set_default_device("cuda:0")
# random.seed(0)
# torch.cuda.manual_seed_all(0)

# doc_len = 1000
# num_doc = 10
# batch_size = 1
# seq_lens = [[20] + [doc_len] * num_doc + [20]] * batch_size
# print(seq_lens)

class DocParialCached:
    def __init__(self, num_doc, doc_len, prefix_suffix_len = 20):
        self.num_doc = num_doc
        self.doc_len = doc_len
        self.prefix_suffix_len = prefix_suffix_len
        self.prompt_len = self.prefix_suffix_len * 2 + self.num_doc * self.doc_len
        self.num_hit_doc = random.randint(1, self.num_doc - 1)
    
    def get_query_lens(self):
        return [self.prefix_suffix_len] + [self.doc_len] * self.num_hit_doc + [self.prefix_suffix_len]

    def get_seq_lens(self):
        return [self.prefix_suffix_len] + [self.doc_len] * self.num_hit_doc + [self.prompt_len]
    
    def get_kvcache_seqused(self):
        return [-1]* (self.num_hit_doc + 1) + [self.prompt_len]
    
    def get_prompt_len(self):
        return self.prompt_len
    
    def get_use_block_table_mask(self):
        return [False]* (self.num_hit_doc + 1) + [True]

prefix_suffix_len = 20
num_doc = 10
batch_size = 10
doc_len = 1000
docs: list[DocParialCached] = []
for _ in range(batch_size):
    docs.append(DocParialCached(num_doc=num_doc, doc_len=doc_len, prefix_suffix_len=prefix_suffix_len))
query_lens = [l for doc in docs for l in doc.get_query_lens()]
prompt_lens = [doc.get_prompt_len() for doc in docs]
max_kv_len = max(prompt_lens)
kvcache_used = [l for doc in docs for l in doc.get_kvcache_seqused()]
use_block_table_mask = [flag for doc in docs for flag in doc.get_use_block_table_mask()]

num_heads = (5, 1)
head_size = 128
dtype = torch.float16
num_blocks = 8192
block_size = 16
fa_version = 2
num_query_heads = num_heads[0]
num_kv_heads = num_heads[1]
window_size = (-1, -1)
scale = head_size**-0.5

query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
local_key = torch.randn(sum(query_lens),
                        num_kv_heads,
                        head_size,
                        dtype=dtype)
local_val = torch.randn_like(local_key)

cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)
cu_query_lens_cpu_list = cu_query_lens.cpu()

key_cache = torch.randn(num_blocks,
                        block_size,
                        num_kv_heads,
                        head_size,
                        dtype=dtype)
value_cache = torch.randn_like(key_cache)
seqused_k = torch.tensor(kvcache_used, dtype=torch.int32)
max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
block_tables = torch.randint(0,
                            num_blocks,
                            (len(query_lens), max_num_blocks_per_seq),
                            dtype=torch.int32)
max_query_len = max(query_lens)
local_cu_kv_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)

print(f"cu_query_lens: {cu_query_lens}")
print(f"max_query_len: {max_query_len}")
print(f"seqused_k: {seqused_k}")
print(f"max_kv_len: {max_kv_len}")
print(f"local_cu_kv_lens: {local_cu_kv_lens}")
output_tmp = torch.zeros_like(query)
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
    local_key=local_key,
    local_value=local_val,
    local_cu_seqlen_k=local_cu_kv_lens,
    out=output_tmp,
    fa_version=fa_version,
)
torch.cuda.synchronize()
# print(output_tmp)
# print(output_tmp)


output = torch.zeros_like(query)
for bidx, flag in enumerate(use_block_table_mask):
    left = cu_query_lens_cpu_list[bidx]
    right = cu_query_lens_cpu_list[bidx + 1]
    kvlen = kvcache_used[bidx]
    if flag:
        assert kvlen != -1
        flash_attn_varlen_func(
            q=query[left : right],
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=torch.tensor([0, right - left], dtype=torch.int32),
            max_seqlen_q=(right - left),
            seqused_k=torch.tensor([kvlen], dtype=torch.int32),
            max_seqlen_k=kvlen,
            softmax_scale=scale,
            causal=True,
            block_table=block_tables[bidx:bidx+1],
            out=output[left : right],
            fa_version=fa_version,
        )
        torch.cuda.synchronize()
    else:
        assert kvlen == -1
        flash_attn_varlen_func(
            q=query[left : right],
            k=local_key[left : right],
            v=local_val[left : right],
            cu_seqlens_q=torch.tensor([0, right - left], dtype=torch.int32),
            max_seqlen_q=(right - left),
            cu_seqlens_k=torch.tensor([0, right - left], dtype=torch.int32),
            max_seqlen_k=(right - left),
            softmax_scale=scale,
            causal=True,
            out=output[left : right],
            fa_version=fa_version,
        )
        torch.cuda.synchronize()
# print(output)
        
print(torch.abs(output - output_tmp).max())
print(torch.allclose(output, output_tmp, atol=1e-3))