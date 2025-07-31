from model.performer.performer_pytorch import PerformerLM
import torch.nn as nn
import torch

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        return x


class Performer(nn.Module):
    def __init__(
        self,
        CLASS = 7,
        SEQ_LEN = 16906,
        POS_EMBED_USING: bool = True,
    ):
        super().__init__()
        self.model = PerformerLM(
            num_tokens=CLASS,
            dim=200,
            depth=6,
            max_seq_len=SEQ_LEN,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=POS_EMBED_USING
        )
        self.model.to_out = Identity()
    def forward(self, x):
        out = self.model(x)
        return out
