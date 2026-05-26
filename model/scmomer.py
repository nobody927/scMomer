import torch
import torch.nn as nn
from model.atac_encoder import ViTModel, MAEConfig
from model.rna_encoder import Performer

def exists(val):
    return val is not None

class scMomer(nn.Module):
    def __init__(
        self,
        config,
        atac_config,
        rna_decoder=None,
        atac_decoder=None,
        sub_task=None,
        encoder=None,
    ):
        super().__init__()
        self.config = config

        if atac_config is not None:
            self.atac_model = ViTModel(atac_config)
            self.atac_projection = nn.Linear(
                self.atac_model.config.hidden_size, config.projection_dim, bias=False
            )

        self.rna_model = Performer()
        self.rna_projection = nn.Linear(
            16906, config.projection_dim, bias=False
        )
        self.to_out = nn.Linear(
            256,128
        )
        self.RNA = rna_decoder
        self.ATAC = atac_decoder
        self.sub_task = sub_task
        self.encoder = encoder
        self.rna_ln = nn.LayerNorm(config.projection_dim)
        self.atac_ln = nn.LayerNorm(config.projection_dim)

    def forward(self, atac_values=None, rna_values=None, reconstruct=False, mode='multimodal'):
        """
        Args:
            atac_values:  ATAC input tensor (required for mode='multimodal')
            rna_values:   RNA input tensor (required for both modes)
            reconstruct:  If True, return decoder outputs (pretrain). If False, return sub_task or embedding.
            mode:         'multimodal' or 'one'

        Returns:
            mode='multimodal', reconstruct=True:   (r_atac, r_rna, logits)  — pretrain_multimodal
            mode='multimodal', reconstruct=False:  atac_embeds              — get_latent
            mode='one',        reconstruct=True:   (r_atac, r_rna, atac_embeds) — pretrain_missing
            mode='one',        reconstruct=False:  sub_task(out)            — translation / celltype
        """
        rna_long = torch.round(rna_values).long().clamp(min=0, max=5)

        if mode == 'multimodal':
            atac_embeds = self._get_atac_features(atac_values)
            rna_embeds = self._get_rna_features(rna_long)
            atac_embeds = self.atac_ln(atac_embeds)
            rna_embeds = self.rna_ln(rna_embeds)

            if reconstruct:
                # Pretrain: return ATAC recon, RNA recon, and discriminator output
                out = self.to_out(torch.cat([atac_embeds, rna_embeds], 1))
                r_atac = self.ATAC(out)
                r_rna = self.RNA(out)
                logits = self.sub_task(out) if exists(self.sub_task) else None
                return r_atac, r_rna, logits
            else:
                # Get latent: return ATAC embedding
                return atac_embeds

        elif mode == 'one':
            # RNA-only: student encoder mimics ATAC embedding
            rna_embeds = self._get_rna_features(rna_long)
            rna_embeds = self.rna_ln(rna_embeds)
            atac_embeds = self.encoder(rna_values)

            out = self.to_out(torch.cat([atac_embeds, rna_embeds], 1))

            if reconstruct:
                # Pretrain missing: return reconstructions + predicted ATAC embedding
                r_atac = self.ATAC(out)
                r_rna = self.RNA(out)
                return r_atac, r_rna, atac_embeds
            else:
                # Downstream: return sub_task output (e.g. celltype logits)
                out = self.sub_task(out)
                return out

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_cell_embedding(self, rna_values):
        """Return 128-dim cell embedding (frozen scMomer + student encoder).

        Used by downstream tasks (drug response, etc.) that need the
        concatenated projection without the sub_task head.

        Args:
            rna_values: RNA input tensor [batch, gene_num]

        Returns:
            cell_embeds: [batch, 128] tensor
        """
        rna_long = torch.round(rna_values).long().clamp(min=0, max=5)
        rna_embeds = self._get_rna_features(rna_long)
        rna_embeds = self.rna_ln(rna_embeds)
        atac_embeds = self.encoder(rna_values)
        out = self.to_out(torch.cat([atac_embeds, rna_embeds], 1))
        return out

    def _get_atac_features(self, atac_values=None):
        atac_outputs = self.atac_model(atac_values)
        atac_features = atac_outputs[1]  # pooled_output
        atac_features = self.atac_projection(atac_features)
        if self.config.normalize:
            atac_features = atac_features / atac_features.norm(dim=-1, keepdim=True)

        return atac_features

    def _get_rna_features(self, rna_values=None):
        rna_outputs = self.rna_model(rna_values)
        rna_features = self.rna_projection(rna_outputs)
        if self.config.normalize:
            rna_features = rna_features / rna_features.norm(dim=-1, keepdim=True)

        return rna_features
