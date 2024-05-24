import torch
from torch import nn
import functools


class XSoftmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        rmask = ~(mask.bool())
        output = input.masked_fill(rmask, float('-inf'))
        output = torch.softmax(output, self.dim)
        output = output.masked_fill(rmask, 0)
        return output


class DebertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class DisentangledSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.in_proj = nn.Linear(config.hidden_size, self.all_head_size * 2, bias=True)
        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.share_att_key = getattr(config, 'share_att_key', False)
        self.pos_att_type = ['c2p', 'p2c']
        self.relative_attention = True

        if self.relative_attention:
            self.position_buckets = 256
            self.max_relative_positions = -1
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads).float()
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads).float()
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

        rel_att = None
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1

        scale = 1/math.sqrt(query_layer.size(-1)*scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2)*scale)
        if self.relative_attention:
            rel_attn = attention_scores.view(-1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1))

        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        

    def disentangled_attention_bias(self, q_layer, k_layer, rel_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size=self.position_buckets,
                                                   max_position=self.max_relative_positions, device=query_layer.device)
