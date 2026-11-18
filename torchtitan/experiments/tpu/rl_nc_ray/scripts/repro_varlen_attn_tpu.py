# Eager traceback:
# RuntimeError: operator 'aten::_flash_attention_forward' is not implemented for TPU. Please file a feature request
#
# Compiled traceback:
# torch._dynamo.exc.BackendCompilerFailed: backend='tpu' raised:
# RuntimeError: operator 'aten::_flash_attention_forward' is not implemented for TPU. Please file a feature request

import torch
import traceback
import sys

try:
    import torch_tpu
    from torch_tpu._loader import load
    load()
    print("Successfully loaded torch_tpu")
except ImportError:
    print("torch_tpu package not found. Make sure you are using the correct virtualenv.")
    sys.exit(1)

def run_repro():
    device = torch.device("tpu")
    print(f"Using device: {device}")

    # Define dimensions for packed sequence attention
    total_tokens = 8
    num_heads = 4
    head_dim = 64

    # Create dummy tensors on TPU
    xq = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    xk = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    xv = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    # Cumulative sequence lengths (offsets) for two packed sequences of lengths 3 and 5
    cu_seq_q = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
    cu_seq_k = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
    max_q = 5
    max_k = 5

    # Import the native PyTorch variable-length attention function
    from torch.nn.attention.varlen import varlen_attn

    print("Attempting to execute varlen_attn eagerly on TPU...")
    try:
        out = varlen_attn(
            xq,
            xk,
            xv,
            cu_seq_q,
            cu_seq_k,
            max_q,
            max_k,
        )
        print(f"SUCCESS (Eager): Result shape is {out.shape}")
    except Exception as e:
        print("\n--- EAGER EXECUTION FAILED ---")
        traceback.print_exc()

    print("\nAttempting to execute torch.compile(varlen_attn) on TPU...")
    @torch.compile(backend="tpu", fullgraph=True)
    def compiled_varlen_attn(q, k, v, cq, ck, mq, mk):
        return varlen_attn(q, k, v, cq, ck, mq, mk)

    try:
        out = compiled_varlen_attn(xq, xk, xv, cu_seq_q, cu_seq_k, max_q, max_k)
        print(f"SUCCESS (Compiled): Result shape is {out.shape}")
    except Exception as e:
        print("\n--- COMPILED EXECUTION FAILED ---")
        traceback.print_exc()

if __name__ == "__main__":
    run_repro()
