import torch
import torch.nn as nn
from ebm.models.base_policy import BasePolicy
from ebm.models.core_transformer import (CoreTransformer,
                                         MLPHead,
                                         SinusoidalPositionalEncoding,
                                         AggregateFeatures)
import copy
import robomimic.utils.tensor_utils as TensorUtils
from ebm.models.encoder_image import R3MEncoder, FineTuneEncoderImage
from ebm.models.encoder_language import FineTuneEncoderLanguage


class EnergyModel(BasePolicy):
    def __init__(self,
                 cfg,
                 shape_meta,
                 embed_size_inp=256,
                 embed_size_cond=256,
                 load_encoded_data=True):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        # Pos Encoding
        input_pos_encoding = copy.deepcopy(policy_cfg.temporal_position_encoding)
        cond_pos_encoding = copy.deepcopy(policy_cfg.temporal_position_encoding)
        input_pos_encoding.network_kwargs.input_size = embed_size_inp
        cond_pos_encoding.network_kwargs.input_size = embed_size_cond
        self.temporal_position_encoding_inp = eval(input_pos_encoding.network)(**input_pos_encoding.network_kwargs)
        self.temporal_position_encoding_cond = eval(cond_pos_encoding.network)(**cond_pos_encoding.network_kwargs)

        # Core Transformer module
        self.core_transformer = CoreTransformer(input_size=embed_size_inp,
                                                input_size_cond=embed_size_cond,
                                                num_layers=policy_cfg.transformer_num_layers,
                                                num_heads=policy_cfg.transformer_num_heads,
                                                head_output_size=policy_cfg.transformer_head_output_size,
                                                mlp_hidden_size=policy_cfg.transformer_mlp_hidden_size,
                                                dropout=policy_cfg.transformer_dropout
                                                )

        self.energy_head = MLPHead(input_size=embed_size_inp,
                                   dropout=policy_cfg.policy_head.dropout)

        self.latent_queue = []
        self.max_seq_len = policy_cfg.transformer_max_seq_len

        self.load_encoded_data = load_encoded_data

        # pretrained image and text encoder
        self.image_encoder_pre = eval(policy_cfg.image_encoder.network)()

        # TODO: Add small MLPs here
        self.image_encoder_finetune = FineTuneEncoderImage(
            in_fts=policy_cfg.image_encoder.network_kwargs.finetune_in_fts,
            out_fts=embed_size_inp)

        self.text_encoder = FineTuneEncoderLanguage(
            in_fts=policy_cfg.language_encoder.network_kwargs.input_size,
            out_fts=embed_size_cond)

        self.proprio_encoder = None

        self.agg_feats = AggregateFeatures(policy_cfg)

        # latent token
        self.embed_size_inp = embed_size_inp

        if policy_cfg.latent_token == "init_random":
            self.latent_token = nn.Parameter(torch.randn(embed_size_inp), requires_grad=True)
        else:
            # from observations
            self.latent_token = self.get_latent_from_observation(policy_cfg.latent_token)

    def get_latent_from_observation(self, init_method):
        raise NotImplementedError

    def encode_inp_cond(self, data):
        encoded_inp = []
        encoded_task = []
        enc_proprio = self.proprio_encoder(data["obs"])
        encoded_inp.append(enc_proprio)  # [B, H, 1, d1]

        B, H = enc_proprio.shape[:2]

        if self.load_encoded_data:
            for img_name in data["img_names"]:
                encoded_img = self.image_encoder(data["encoded_img"][img_name])
                encoded_inp.append(encoded_img)  # [B, H, 1, d2]

            for task_name in data["task_names"]:
                encoded_task = self.text_encoder(data["encoded_task"][task_name])  # [B, d3]
                encoded_task = encoded_task.view(B, 1, 1, -1).expand(-1, H, -1, -1)
        else:
            raise NotImplementedError

        # adding latent tokens to the encoded input
        expanded_latent_token = self.latent_token.view(1, 1, 1, self.embed_size_inp).expand(B, H, 1, self.embed_size_inp)
        encoded_inp.append(expanded_latent_token)

        # concatenating over modality token dimension
        encoded_inp = torch.cat(encoded_inp, dim=-2)
        return encoded_inp, encoded_task

    def core_transformer_decoder(self, x, c):
        # x -> (B, H, num_modality, E)

        pos_emb_inp = self.temporal_position_encoding_inp(x)
        pos_emb_cond = self.temporal_position_encoding_cond(c)

        x = x + pos_emb_inp.unsqueeze(1)
        c = c + pos_emb_cond.unsqueeze(1)

        shape_inp = x.shape

        x = TensorUtils.join_dimensions(x, 1, 2)  # [B, H*N, d]
        c = TensorUtils.join_dimensions(c, 1, 2)  # [B, H*N, dc]

        # Todo: Experiment with masks, with and without history
        x = self.core_transformer(x, c, None)

        # Reshaping to input shape
        x = x.reshape(*shape_inp)
        # last modality is for latent vector -> parameter for dmp
        latent_out = x[:, :, -1]
        agg_latent = self.agg_feats(latent_out)
        out = self.energy_head(agg_latent)

        return out

    def forward(self, data):
        data = self.preprocess_input(data, train_mode=True)
        x, c = self.encode_inp_cond(data)
        energy = self.core_transformer_decoder(x, c)
        return energy

    def get_energy(self, data):
        # Todo: Implement a latent queue for history
        raise NotImplementedError
        # self.eval()
        # with torch.no_grad():
        #     data = self.preprocess_input(data, train_mode=False)
        #     x, c = self.encode_inp_cond(data)
        #     energy = self.core_transformer_decoder(x, c)



