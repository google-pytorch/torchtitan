from absl.testing import absltest
import torch
from torchtitan.experiments.tpu.conformer import model


class _ReferenceMHA(torch.nn.Module):
  """Reference Multi-Head Attention implementation.

  This represents the implementation before cl/922374432, used to verify
  mathematical equivalence of the optimized PatchedMHA.
  """

  def __init__(self, embed_dim, num_heads, dropout=0.0):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout = dropout
    self.head_dim = embed_dim // num_heads
    self.in_proj = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=True)
    self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)

  def forward(
      self,
      query,
      key,
      value,
      key_padding_mask=None,
      need_weights=True,
      attn_mask=None,
      average_attn_weights=True,
      is_causal=False,
  ):
    bsz, tgt_len, hidden_dim = query.shape
    w = self.in_proj.weight.view(3, self.num_heads, self.head_dim, hidden_dim)
    qkv = torch.einsum("btd,nhkd->nbhtk", query, w)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    w_out = self.out_proj.weight.view(
        self.out_proj.out_features, self.num_heads, self.head_dim
    )
    outputs = []
    for i in range(self.num_heads):
      x_i = attn_output[:, i, :, :]
      w_i = w_out[:, i, :].t().unsqueeze(0).expand(bsz, -1, -1)
      head_out = torch.bmm(x_i, w_i)
      outputs.append(head_out)
    attn_output = sum(outputs)
    if self.out_proj.bias is not None:
      attn_output = attn_output + self.out_proj.bias
    return attn_output, None


class ConformerModelTest(absltest.TestCase):

  def test_instantiation_and_forward(self):
    args = model.ConformerModelArgs(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        kernel_size=31,
    )

    conformer_model = model.Conformer(args)

    batch_size = 2
    seq_len = 16
    inputs = torch.randint(0, args.vocab_size, (batch_size, seq_len), dtype=torch.long)

    logits = conformer_model(inputs)

    self.assertEqual(logits.shape, (batch_size, seq_len, args.vocab_size))
    print("Forward pass successful. Shape:", logits.shape)

  def test_patched_mha_equivalence(self):
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 16

    ref_mha = _ReferenceMHA(embed_dim=embed_dim, num_heads=num_heads)
    opt_mha = model.PatchedMHA(embed_dim=embed_dim, num_heads=num_heads)

    # Copy weights
    opt_mha.in_proj.weight.data.copy_(ref_mha.in_proj.weight.data)
    opt_mha.in_proj.bias.data.copy_(ref_mha.in_proj.bias.data)
    opt_mha.out_proj.weight.data.copy_(ref_mha.out_proj.weight.data)
    opt_mha.out_proj.bias.data.copy_(ref_mha.out_proj.bias.data)

    query = torch.randn(batch_size, seq_len, embed_dim)

    ref_out, _ = ref_mha(query, query, query)
    opt_out, _ = opt_mha(query, query, query)

    torch.testing.assert_close(opt_out, ref_out)


if __name__ == "__main__":
  absltest.main()
