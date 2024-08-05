import torch
import torch.nn as nn

from galerkin_transformer.model import SimpleAttention
from positionEmbedding import PositionEmbeddingSine
from ffn import FFNLayer

class SelfAttention(nn.Module):
    def __init__(self, n_feats=64 ):
        super(SelfAttention, self).__init__()
        _attention_types = [
            "linear",
            "galerkin",
            "global",
            "causal",
            "fourier",
            "softmax",
            "integral",
            "local",
        ]
        self.n_feats = n_feats

        _norm_types = ["instance", "layer"]
        norm_type = _norm_types[1]
        attn_norm = True
        n_head = 8
        dropout = 0.1
        self.sa = SimpleAttention(
            n_head=n_head,
            d_model=n_feats,
            attention_type=_attention_types[1],
            pos_dim=-1,
            norm=attn_norm,
            norm_type=norm_type,
            dropout=0.0,
        )
        self.posEmbedding = PositionEmbeddingSine(
            num_pos_feats=n_feats // 2, normalize=True
        )
        self.dropout = nn.Dropout(dropout)

        self.layerNorm = nn.LayerNorm(n_feats)
        
        self.ffn = FFNLayer(n_feats , n_feats * 2 , normalize_before= True)
        
    def forward(self , x  ,  pos = None):
        if(len(x.shape) == 3):
            return self.tensorForward(x)
        elif(len(x.shape) == 4):
            return self.imageForward(x)
        else :
            raise("输入了啥看不懂")
        
    # x is supposed to be in shape of [ b , c , h  , w]
    def imageForward(self, x , pos = None):
        b, c, h, w = x.shape

        assert c == self.n_feats , "输入数据的维度与预期不一致！"
        # at first ,transpose x to [b , h*w , c]
        if (pos is None):
            pos = self.posEmbedding(x, None).permute(0, 2, 3, 1).contiguous().view(b, -1, c)
            
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        # attention
        x, _ = self.sa(query=x + pos, key=x + pos, value=x)

        x = x + self.dropout(x)

        x = self.ffn(x)

        x = x + self.dropout(x)
        # x = self.layerNorm(x)

        # transpose x back to [ b , c , h  , w]
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, -1)
        return x

    def tensorForward(self , x ):
        b, n , c = x.shape

        assert c == self.n_feats , "输入数据的维度与预期不一致！"
        # # at first ,transpose x to [b , h*w , c]
        # if (pos is None):
        #     pos = self.posEmbedding(x, None).permute(0, 2, 3, 1).contiguous().view(b, -1, c)
            
        # x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        
        # attention
        x_, _ = self.sa(query=x , key=x , value=x)
        
        if(torch.isnan(x_).any()):
            print("nan value detect after galerkin  self attention ")

        x_ = x /2  + self.dropout(x_) /2

        x_ = self.ffn(x_)

        x = x / 2 + self.dropout(x_) / 2
        # x = self.layerNorm(x)

        # # transpose x back to [ b , c , h  , w]
        # x = x.permute(0, 2, 1).contiguous().view(b, c, h, -1)
        return x
