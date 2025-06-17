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
from ebm.models.encoder_proprio import EncoderProprio
from ebm.models.dmp_head import DMPHead


class EnergyModel(BasePolicy):
    def __init__(self,
                 cfg,
                 shape_meta,
                 batch_size,
                 embed_size_inp=256,
                 embed_size_cond=256,
                 load_encoded_data=True):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy
        self.cfg = cfg

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

        self.static_image_encoder_finetune = FineTuneEncoderImage(
            in_fts=policy_cfg.image_encoder.network_kwargs.finetune_in_fts,
            out_fts=embed_size_inp
        )

        if cfg.encode_gripper_cam:
            self.gripper_image_encoder_finetune = FineTuneEncoderImage(
                in_fts=policy_cfg.image_encoder.network_kwargs.finetune_in_fts,
                out_fts=embed_size_inp
            )

        self.text_encoder = FineTuneEncoderLanguage(
            in_fts=policy_cfg.language_encoder.network_kwargs.input_size,
            out_fts=embed_size_cond
        )

        self.proprio_encoder = EncoderProprio(policy_cfg.proprio_encoder.proprio_encoder_type,
                                              use_joint=policy_cfg.use_joint,
                                              use_gripper=policy_cfg.use_gripper,
                                              use_ee=policy_cfg.use_ee,
                                              out_dim=embed_size_inp,
                                              dropout=policy_cfg.proprio_encoder.dropout
                                              )

        dmp_out_size = self.cfg.motion_primitives.num_basis_fns * self.proprio_encoder.get_proprio_size() + \
                       self.proprio_encoder.get_proprio_size()

        self.dmp_head = DMPHead(input_size=embed_size_inp,
                                output_size=dmp_out_size,
                                dropout=policy_cfg.policy_head.dropout)

        self.agg_feats = AggregateFeatures(policy_cfg)

        # latent token
        self.embed_size_inp = embed_size_inp
        self.batch_size = batch_size

    def get_latent_from_observation(self, init_method):
        raise NotImplementedError

    def get_inputs_to_model(self, data):
        # proprio input
        input_proprio = self.proprio_encoder.get_input_data(data['obs'])

        # static image input
        image_static = data['obs']['agentview_rgb']  # [B, H, c, h, w]
        B, H, C, He, Wi = image_static.shape
        with torch.no_grad():
            encoded_static_image_pre = self.image_encoder_pre(image_static.reshape(B * H, C, He, Wi))

        # gripper cam image input
        if self.cfg.encode_gripper_cam:
            image_gripper = data['obs']['eye_in_hand_rgb']
            B, H, C, He, Wi = image_gripper.shape
            with torch.no_grad():
                encoded_gripper_image_pre = self.image_encoder_pre(image_gripper.reshape(B*H, C, He, Wi))

        if self.cfg.encode_gripper_cam:
            return input_proprio, encoded_static_image_pre, encoded_gripper_image_pre
        else:
            return input_proprio, encoded_static_image_pre


    def encode_inp_cond(self,
                        data,
                        latent_token,
                        inp_proprio,  # [B, H, dims]
                        encoded_static_image_pre,  # [B*H, 2048]
                        encoded_gripper_image_pre,
                        encode_task=True):  # [B*H, 2048]
        encoded_inp = []

        # encode proprioceptive observation
        enc_proprio = self.proprio_encoder(inp_proprio)
        encoded_inp.append(enc_proprio)  # [B, H, 1, d1]

        # encode static image
        image_static = data['obs']['agentview_rgb'] # [B, H, c, h, w]
        B, H, C, He, Wi = image_static.shape
        encoded_static_image = self.static_image_encoder_finetune(encoded_static_image_pre)
        encoded_static_image = encoded_static_image.reshape(B, H, 1, -1)
        encoded_inp.append(encoded_static_image)

        # encode gripper image
        if self.cfg.encode_gripper_cam:
            image_gripper = data['obs']['eye_in_hand_rgb']
            B, H, C, He, Wi = image_gripper.shape
            encoded_gripper_image = self.gripper_image_encoder_finetune(encoded_gripper_image_pre)
            encoded_gripper_image = encoded_gripper_image.reshape(B, H, 1, -1)
            encoded_inp.append(encoded_gripper_image)

        # encode text
        if encode_task:
            encoded_text = self.text_encoder(data["task_emb"].to("cuda")) # [B, dim]
            encoded_text = encoded_text.view(B, 1, 1, -1).expand(-1, H, -1, -1) # [B, H, 1, dim]
        else:
            encoded_text = None

        # adding latent tokens to the encoded input
        expanded_latent_token = latent_token.view(B, 1, 1, self.embed_size_inp).expand(B, H, 1, self.embed_size_inp)
        encoded_inp.append(expanded_latent_token)

        # concatenating over modality token dimension
        encoded_inp = torch.cat(encoded_inp, dim=-2)

        return encoded_inp, encoded_text

    def mcmc_sampling_all(self,
                          num_steps,
                          data,
                          inp_proprio,
                          encoded_static_image_pre,
                          encoded_gripper_image_pre,
                          latent_token,
                          step_sizes,  # [proprio, static_img, gripper_img, latent]
                          noise_scales,
                          clip_grad_norm=None  # 1.0
                          ):

        # past observations
        B, H = inp_proprio.shape[:2]
        past_inp_proprio = inp_proprio[:, :-1]
        encoded_static_image_pre = encoded_static_image_pre.reshape(B, H, -1)
        encoded_gripper_image_pre = encoded_gripper_image_pre.reshape(B, H, -1)
        past_img_static = encoded_static_image_pre[:, :-1]
        past_img_gripper = encoded_gripper_image_pre[:, :-1]

        # current observations
        curr_inp_proprio = inp_proprio[:, -1].clone().detach().requires_grad_(True).unsqueeze(1)
        curr_img_static = encoded_static_image_pre[:, -1].clone().detach().requires_grad_(True).unsqueeze(1)
        if self.cfg.encode_gripper_cam:
            curr_img_gripper = encoded_gripper_image_pre[:, -1].clone().detach().requires_grad_(True).unsqueeze(1)
        latent_token = latent_token.clone().detach().requires_grad_(True)

        for _ in range(num_steps):
            # setting req grads to True as we detach in the last step of this loop
            curr_inp_proprio.requires_grad_(True)
            curr_img_static.requires_grad_(True)
            if self.cfg.encode_gripper_cam:
                curr_img_gripper.requires_grad_(True)
            latent_token.requires_grad_(True)

            # reconstruct input data
            full_inp_proprio = torch.cat([past_inp_proprio, curr_inp_proprio], dim=1).to(inp_proprio.device)
            full_img_static = torch.cat([past_img_static, curr_img_static], dim=1).to(inp_proprio.device)
            if self.cfg.encode_gripper_cam:
                full_img_gripper = torch.cat([past_img_gripper, curr_img_gripper], dim=1).to(inp_proprio.device)
            else:
                full_img_gripper = None

            # reshaping to match required dimension for the energy function
            full_img_static = full_img_static.reshape(B*H, -1)
            if self.cfg.encode_gripper_cam:
                full_img_gripper = full_img_gripper.reshape(B*H, -1)

            # compute energy
            x, c = self.encode_inp_cond(data, latent_token, full_inp_proprio, full_img_static, full_img_gripper)
            energy, _ = self.core_transformer_decoder(x, c)  # scalar energy
            energy = energy.sum()  # Making it scalar for grads to flow

            # compute gradients
            if self.cfg.encode_gripper_cam:
                grad_inp_proprio, grad_inp_static, grad_inp_gripper, grad_inp_latent = torch.autograd.grad(
                    energy, (curr_inp_proprio, curr_img_static, curr_img_gripper, latent_token)
                )
            else:
                grad_inp_proprio, grad_inp_static, grad_inp_latent = torch.autograd.grad(
                    energy, (curr_inp_proprio, curr_img_static, latent_token)
                )

            # Gradient Clipping for stability
            if clip_grad_norm is not None:
                grad_inp_proprio = torch.nn.utils.clip_grad_norm_(grad_inp_proprio, clip_grad_norm)
                grad_inp_static = torch.nn.utils.clip_grad_norm_(grad_inp_static, clip_grad_norm)
                if self.cfg.encode_gripper_cam:
                    grad_inp_gripper = torch.nn.utils.clip_grad_norm_(grad_inp_gripper, clip_grad_norm)
                grad_inp_latent = torch.nn.utils.clip_grad_norm_(grad_inp_latent, clip_grad_norm)

            curr_inp_proprio = (curr_inp_proprio - (0.5 * step_sizes[0] * grad_inp_proprio) +
                                (noise_scales[0] * torch.randn_like(curr_inp_proprio)))

            curr_img_static = (curr_img_static - (0.5 * step_sizes[1] * grad_inp_static) +
                               (noise_scales[1] * torch.randn_like(curr_img_static)))

            if self.cfg.encode_gripper_cam:
                curr_img_gripper = (curr_img_gripper - (0.5 * step_sizes[2] * grad_inp_gripper) +
                                    (noise_scales[2] * torch.randn_like(curr_img_gripper)))

            latent_token = (latent_token - (0.5 * step_sizes[3] * grad_inp_latent) +
                            (noise_scales[3] * torch.randn_like(latent_token)))

            # detaching gradients
            curr_inp_proprio = curr_inp_proprio.detach()
            curr_img_static = curr_img_static.detach()
            if self.cfg.encode_gripper_cam:
                curr_img_gripper = curr_img_gripper.detach()
            latent_token = latent_token.detach()

        full_inp_proprio = torch.cat([past_inp_proprio, curr_inp_proprio], dim=1).to(inp_proprio.device)
        full_img_static = torch.cat([past_img_static, curr_img_static], dim=1).to(inp_proprio.device)
        if self.cfg.encode_gripper_cam:
            full_img_gripper = torch.cat([past_img_gripper, curr_img_gripper], dim=1).to(inp_proprio.device)
        else:
            full_img_gripper = None

        full_img_static = full_img_static.reshape(B * H, -1)
        if self.cfg.encode_gripper_cam:
            full_img_gripper = full_img_gripper.reshape(B * H, -1)

        if self.cfg.encode_gripper_cam:
            return full_inp_proprio, full_img_static, full_img_gripper, latent_token
        else:
            return full_inp_proprio, full_img_static, latent_token

    def mcmc_sampling_latent(self,
                             num_steps,
                             data,
                             inp_proprio,
                             encoded_static_image_pre,
                             encoded_gripper_image_pre,
                             latent_token,
                             step_size,  # [proprio, static_img, gripper_img, latent]
                             noise_scale,
                             clip_grad_norm=None  # 1.0
                             ):

        # current observations
        latent_token = latent_token.clone().detach().requires_grad_(True)
        for _ in range(num_steps):
            # compute energy
            x, c = self.encode_inp_cond(data,
                                        latent_token,
                                        inp_proprio,
                                        encoded_static_image_pre,
                                        encoded_gripper_image_pre)
            energy, _ = self.core_transformer_decoder(x, c)  # scalar energy
            energy = energy.sum()  # Making it scalar

            # compute gradients
            grad_inp_latent = torch.autograd.grad(energy, latent_token)

            # Gradient Clipping for stability
            if clip_grad_norm is not None:
                grad_inp_latent = torch.nn.utils.clip_grad_norm_(grad_inp_latent, clip_grad_norm)

            latent_token = (latent_token - (0.5 * step_size * grad_inp_latent) +
                            (noise_scale * torch.randn_like(latent_token)))

            # detaching gradients
            latent_token = latent_token.detach()

            return latent_token

    def local_negative_sampling(self,
                                data,
                                inp_proprio,
                                encoded_static_image_pre,
                                encoded_gripper_image_pre,
                                latent_token
                                ):
        step_sizes = [self.cfg.sampling.step_size_proprio,
                      self.cfg.sampling.step_size_static_img,
                      self.cfg.sampling.step_size_gripper_img,
                      self.cfg.sampling.step_size_latent]

        noise_scales = [self.cfg.sampling.noise_scale_proprio,
                        self.cfg.sampling.noise_scale_static_img,
                        self.cfg.sampling.noise_scale_gripper_img,
                        self.cfg.sampling.noise_scale_latent]

        neg_samples = self.mcmc_sampling_all(num_steps=self.cfg.sampling.num_steps_neg,
                                             data=data,
                                             inp_proprio=inp_proprio,
                                             encoded_static_image_pre=encoded_static_image_pre,
                                             encoded_gripper_image_pre=encoded_gripper_image_pre,
                                             latent_token=latent_token,
                                             step_sizes=step_sizes,
                                             noise_scales=noise_scales,
                                             clip_grad_norm=self.cfg.sampling.clip_grad_norm)

        return neg_samples

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

        return out, agg_latent

    def forward(self, data):
        data = self.preprocess_input(data, train_mode=True)
        if self.cfg.encode_gripper_cam:
            inp_proprio, encoded_static_image_pre, encoded_gripper_image_pre = self.get_inputs_to_model(data)
        else:
            inp_proprio, encoded_static_image_pre = self.get_inputs_to_model(data)
            encoded_gripper_image_pre = None

        if self.cfg.policy.latent_token == "init_random":
            latent_token = torch.randn(self.batch_size,
                                       self.embed_size_inp,
                                       device="cuda",
                                       requires_grad=True)
        else:
            # from observations
            latent_token = self.get_latent_from_observation(self.cfg.policy.latent_token)

        latent_token = self.mcmc_sampling_latent(num_steps=self.cfg.sampling.num_steps,
                                                 data=data,
                                                 inp_proprio=inp_proprio,
                                                 encoded_static_image_pre=encoded_static_image_pre,
                                                 encoded_gripper_image_pre=encoded_gripper_image_pre,
                                                 latent_token=latent_token,
                                                 step_size=self.cfg.sampling.step_size_latent,
                                                 noise_scale=0,  # self.cfg.sampling.noise_scale_latent,
                                                 clip_grad_norm=self.cfg.sampling.clip_grad_norm)

        x, c = self.encode_inp_cond(data,
                                    latent_token,
                                    inp_proprio,
                                    encoded_static_image_pre,
                                    encoded_gripper_image_pre)

        energy, latent_agg = self.core_transformer_decoder(x, c)

        # dmp head
        dmp_params = self.dmp_head(latent_agg)
        weights = dmp_params[:, : self.cfg.motion_primitives.num_basis_fns * self.proprio_encoder.get_proprio_size()]
        goal = dmp_params[:, self.cfg.motion_primitives.num_basis_fns * self.proprio_encoder.get_proprio_size():]

        # -ve sampling
        neg_samples = self.local_negative_sampling(data,
                                                   inp_proprio,
                                                   encoded_static_image_pre,
                                                   encoded_gripper_image_pre,
                                                   latent_token)
        if self.cfg.encode_gripper_cam:
            neg_inp_proprio, neg_img_static, neg_img_gripper, neg_latent_token = neg_samples
        else:
            neg_inp_proprio, neg_img_static, neg_latent_token = neg_samples
            neg_img_gripper = None

        # compute local -ve energy, for local CD loss
        x, c = self.encode_inp_cond(data,
                                    neg_latent_token,
                                    neg_inp_proprio,
                                    neg_img_static,
                                    neg_img_gripper)

        energy_neg, latent_agg_neg = self.core_transformer_decoder(x, c)

        return energy, weights, goal, energy_neg, latent_token

    def get_global_neg_energy_loss(self,
                                   data_curr,
                                   data_other,
                                   latent_token_curr,):
        # Data for the current task
        curr_data = self.preprocess_input(data_curr, train_mode=True)
        if self.cfg.encode_gripper_cam:
            curr_inp_proprio, curr_encoded_static_image_pre, curr_encoded_gripper_image_pre = self.get_inputs_to_model(curr_data)
        else:
            curr_inp_proprio, curr_encoded_static_image_pre = self.get_inputs_to_model(curr_data)
            curr_encoded_gripper_image_pre = None

        # Data for some other task
        diff_data = self.preprocess_input(data_other, train_mode=True)
        if self.cfg.encode_gripper_cam:
            diff_inp_proprio, diff_encoded_static_image_pre, diff_encoded_gripper_image_pre = self.get_inputs_to_model(curr_data)
        else:
            diff_inp_proprio, diff_encoded_static_image_pre = self.get_inputs_to_model(curr_data)
            diff_encoded_gripper_image_pre = None


        if self.cfg.policy.latent_token == "init_random":
            diff_latent_token = torch.randn(self.batch_size,
                                            self.embed_size_inp,
                                            device="cuda",
                                            requires_grad=True)
        else:
            # from observations
            diff_latent_token = self.get_latent_from_observation(self.cfg.policy.latent_token)


        step_size_all = [self.cfg.sampling.step_size_proprio,
                         self.cfg.sampling.step_size_static_img,
                         self.cfg.sampling.step_size_gripper_img,
                         self.cfg.sampling.step_size_latent]

        noise_scale_all = [self.cfg.sampling.noise_scale_proprio,
                           self.cfg.sampling.noise_scale_static_img,
                           self.cfg.sampling.noise_scale_gripper_img,
                           self.cfg.sampling.noise_scale_latent]

        diff_latent_token = self.mcmc_sampling_latent(num_steps=self.cfg.sampling.num_steps,
                                                      data=diff_data,
                                                      inp_proprio=diff_inp_proprio,
                                                      encoded_static_image_pre=diff_encoded_static_image_pre,
                                                      encoded_gripper_image_pre=diff_encoded_gripper_image_pre,
                                                      latent_token=diff_latent_token,
                                                      step_size=self.cfg.sampling.step_size_latent,
                                                      noise_scale=0,  # self.cfg.sampling.noise_scale_latent,
                                                      clip_grad_norm=self.cfg.sampling.clip_grad_norm)

        # compute encoded inputs
        curr_x, curr_c = self.encode_inp_cond(curr_data,
                                              latent_token_curr,
                                              curr_inp_proprio,
                                              curr_encoded_static_image_pre,
                                              curr_encoded_gripper_image_pre)

        diff_x, diff_c = self.encode_inp_cond(diff_data,
                                              diff_latent_token,
                                              diff_inp_proprio,
                                              diff_encoded_static_image_pre,
                                              diff_encoded_gripper_image_pre)

        # compute cross energies
        energy_cross_sample, latent_agg_neg_cs = self.core_transformer_decoder(diff_x, curr_c)
        energy_cross_task, latent_agg_neg_ct = self.core_transformer_decoder(curr_x, diff_c)

        return energy_cross_sample, energy_cross_task

    def get_energy(self, data):
        # Todo: Implement a latent queue for history
        raise NotImplementedError
        # self.eval()
        # with torch.no_grad():
        #     data = self.preprocess_input(data, train_mode=False)
        #     x, c = self.encode_inp_cond(data)
        #     energy = self.core_transformer_decoder(x, c)



