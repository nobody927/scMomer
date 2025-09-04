import torch
import torch.nn as nn
from model.atac_model import ViTModel, MAEConfig
from model.rna_encoder import Performer
import argparse
import pickle
import os

def exists(val):
    return val is not None

class scMomer(nn.Module):
    def __init__(
        self,
        config,
        atac_config,
        rna_modal=Performer(),
        rna_decoder=None,
        atac_decoder=None,
        sub_task=None,
        encoder= None,
        # rna_config,
    ):
        super().__init__()
        self.config = config
        # self.save_hyperparameters()
        if atac_config != None:
            self.atac_model = ViTModel(atac_config)
            self.atac_projection = nn.Linear(
                self.atac_model.config.hidden_size, config.projection_dim, bias=False
            )
        self.rna_model = rna_modal

        # config.hidden_size = self.atac_model.config.hidden_size

        self.rna_projection = nn.Linear(
            16906, config.projection_dim, bias=False
        )
        self.to_out = nn.Linear(
            256,128, bias=False
        )
        self.RNA = rna_decoder
        self.ATAC = atac_decoder
        self.sub_task = sub_task
        self.encoder = encoder
        # print(f"atac_num_patches: {self.atac_model.embeddings.num_patches}", flush=True)
        # print(f"rna_num_patches: {self.rna_model.embeddings.num_patches}", flush=True)

    # encoder:辅助网络，用于预测缺失模态
    def forward(self, atac_values=None, rna_values=None, reconstruct=True, encoder=None, atac_mean=None, mode='multimodal', get_distill=False):
        rna_long = torch.round(rna_values).long()
        if mode == 'multimodal':
            atac_embeds = self._get_atac_features(atac_values)
            if get_distill:
                return atac_embeds
            rna_embeds = self._get_rna_features(rna_long)
            if exists(self.to_out):
                if reconstruct == True:
                    out = self.to_out(torch.cat([atac_embeds, rna_embeds], 1))
                    RNA = self.RNA(out)
                    ATAC = self.ATAC(out)
                    if self.sub_task != None:
                        logits = self.sub_task(out)
                        return ATAC, RNA, logits
                    return ATAC, RNA
                out = self.to_out(torch.cat([atac_embeds, rna_embeds], 1))
                out = self.sub_task(out)
                return out
        elif mode == 'one':
            # assert encoder is not None
            rna_embeds = self._get_rna_features(rna_long)
            # return rna_embeds
            atac_embeds = self.encoder(rna_values)
            if exists(self.to_out):
                if reconstruct == True:
                    out = self.to_out(torch.cat([atac_embeds, rna_embeds], 1))
                    # return out
                    RNA = self.RNA(out)
                    ATAC = self.ATAC(out)
                    return ATAC, RNA, atac_embeds
                out1 = self.to_out(torch.cat([atac_embeds, rna_embeds], 1))
                # return out1
                # return atac_embeds, rna_embeds, out1
                out = self.sub_task(out1)
                return out
        else:
            raise ValueError('mode selected error')

        return atac_embeds, rna_embeds

    def _get_atac_features(self, atac_values=None):
        atac_outputs = self.atac_model(atac_values)

        atac_features = atac_outputs[1]  # pooled_output
        atac_features = self.atac_projection(atac_features)

        if self.config.normalize:
            atac_features = atac_features / atac_features.norm(dim=-1, keepdim=True)

        return atac_features

    def _get_rna_features(self, rna_values=None):
        rna_outputs = self.rna_model(rna_values)
        # return rna_outputs
        # rna_features = rna_outputs[1]
        rna_features = self.rna_projection(rna_outputs)

        if self.config.normalize:
            rna_features = rna_features / rna_features.norm(dim=-1, keepdim=True)

        return rna_features

    def save_intermediate_outputs(self, atac_values, rna_values, batch_idx, save_dir, batch_data):
        """
        保存当前 batch 的中间输出到同一个文件中
        :param atac_values: ATAC 数据
        :param rna_values: RNA 数据
        :param epoch: 当前 epoch
        :param batch_idx: 当前 batch 索引
        :param save_dir: 保存目录
        """
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 获取中间输出
        atac_embeds = self._get_atac_features(atac_values)
        rna_embeds = self._get_rna_features(rna_values)
        out = self.to_out(torch.cat([atac_embeds, rna_embeds], 1)).detach().cpu()


        batch_data.append({
            'batch_idx': batch_idx,
            'atac_embeds': atac_embeds,
            'rna_embeds': rna_embeds,
            'out': out
        })
        # return batch_data
        # 保存路径
        # save_path = os.path.join(save_dir, f'intermediate_outputs_multi.pkl')
        #
        # # 如果文件不存在，初始化一个空列表
        # if not os.path.exists(save_path):
        #     with open(save_path, 'wb') as f:
        #         pickle.dump([], f)
        #
        # # 加载已保存的数据
        # with open(save_path, 'rb') as f:
        #     saved_data = pickle.load(f)
        #
        # # 追加当前 batch 的中间输出
        # saved_data.append({
        #     'batch_idx': batch_idx,
        #     'atac_embeds': atac_embeds,
        #     'rna_embeds': rna_embeds,
        #     'out': out
        # })
        #
        # # 保存更新后的数据
        # with open(save_path, 'wb') as f:
        #     pickle.dump(saved_data, f)
    def save_batch_data(self, save_dir, batch_data, batch_idx):
        """
        保存累积的中间输出到磁盘
        :param save_dir: 保存目录
        :param batch_data: 累积的中间输出数据
        :param batch_idx: 当前 batch 索引
        """
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 保存路径
        save_path = os.path.join(save_dir, 'intermediate_outputs_multi_val.pkl')
        if not os.path.exists(save_path):
            saved_data = []
        else:
            # 加载已保存的数据
            with open(save_path, 'rb') as f:
                saved_data = pickle.load(f)
        saved_data.append(batch_data)

        # 保存更新后的数据
        with open(save_path, 'wb') as f:
            pickle.dump(saved_data, f)

        # 清空当前 batch 的数据结构
        batch_data.clear()
