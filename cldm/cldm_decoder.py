import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Upsample, AttentionBlock,normalization
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

class ControlledUnetModel_f2i(UNetModel):
    #本质上ControlNet也是一个Unet
    #只有middle_block加入
    def forward(self, x, timesteps=None, context=None, control=None, **kwargs):
        hs = []
        # print(control[0].shape)
        # print(control[0])
        # exit()
        with torch.no_grad():
            #时间嵌入
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                # 如果不加context 在sd的context部分会出问题
                h = module(h, emb, context)
                hs.append(h)

            h = self.middle_block(h, emb, context)
        # 这一步是middle_block加control的处理 将其他与sd的联系全部去除了
        if control is not None:
            h += control[0]
            del control[0]
        for i, (control_part,module) in enumerate(zip(control,self.output_blocks)):
            # torch.Size([4, 1280, 8, 8]) torch.Size([4, 1280, 8, 8])
            # torch.Size([4, 1280, 8, 8]) torch.Size([4, 1280, 8, 8])
            # torch.Size([4, 1280, 8, 8]) torch.Size([4, 1280, 16, 16])
            # torch.Size([4, 1280, 16, 16]) torch.Size([4, 1280, 16, 16])
            # torch.Size([4, 1280, 16, 16]) torch.Size([4, 1280, 16, 16])
            # torch.Size([4, 1280, 16, 16]) torch.Size([4, 1280, 32, 32])
            # torch.Size([4, 1280, 32, 32]) torch.Size([4, 640, 32, 32])
            # torch.Size([4, 640, 32, 32]) torch.Size([4, 640, 32, 32])
            # torch.Size([4, 640, 32, 32]) torch.Size([4, 640, 64, 64])
            # torch.Size([4, 640, 64, 64]) torch.Size([4, 320, 64, 64])
            # torch.Size([4, 320, 64, 64]) torch.Size([4, 320, 64, 64])
            # torch.Size([4, 320, 64, 64]) torch.Size([4, 320, 64, 64])
            h = torch.cat([h , hs.pop()], dim=1)
            h = module(h, emb, context)
            h += control_part
        h = h.type(x.dtype)
        return self.out(h)
class ControlNet_f2i(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        model_channels,
        out_channels,
        feature_dim,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.feature_dim = feature_dim
        self.dims = dims
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.zero_convs = nn.ModuleList()
        input_block_chans = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
        self.feature_up_sample = nn.Sequential(nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(feature_dim),
                    nn.SiLU(),
                    nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=2, stride=1, padding=0),
                    nn.BatchNorm2d(feature_dim),
                    nn.SiLU(),
                    nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=4, stride=1, padding=0),
                    nn.BatchNorm2d(feature_dim),
                    nn.SiLU(),
                    nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=4, stride=1, padding=0),
                    nn.BatchNorm2d(feature_dim),
                    nn.SiLU(),
                    nn.ConvTranspose2d(in_channels=self.feature_dim, out_channels=1280, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(1280),
                    nn.SiLU()
        )
        ch = 1280
        dim_head = 160
        self._feature_size = 10880
        ds = 8
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(1280)
        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        #channel_mult = [1,2,4,4]

        for level, mult in list(enumerate(channel_mult))[::-1]:
            # self.num_res_blocks = [2,2,2,2]
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch ,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, timesteps, context, **kwargs):
        #此时x是moco_feature 向量
        # print(x.shape)
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        outs = []
        h = x.type(self.dtype)

        h = self.feature_up_sample(h)
        h = self.middle_block(h, emb, context)
        #h (bs,1280,8,8)
        outs.append(self.middle_block_out(h, emb, context))
        for module, zero_conv in zip(self.output_blocks, self.zero_convs):
            #zero_conv 1280*5 -> 640*3-> 320*3
            # h shape torch.Size([4, 1280, 8, 8])
            # torch.Size([4, 1280, 8, 8])
            # torch.Size([4, 1280, 16, 16])
            # torch.Size([4, 1280, 16, 16])
            # torch.Size([4, 1280, 16, 16])
            # torch.Size([4, 1280, 32, 32])
            # torch.Size([4, 640, 32, 32])
            # torch.Size([4, 640, 32, 32])
            # torch.Size([4, 640, 64, 64])
            # torch.Size([4, 320, 64, 64])
            # torch.Size([4, 320, 64, 64])
            # torch.Size([4, 320, 64, 64])
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))
        return outs


class ControlLDM_f2i(LatentDiffusion):

    def __init__(self, control_stage_config , control_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_scales = [1.0] * 13
        self.control_key = control_key

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        #如果没有文本 c得不到 令配置中的force_null_conditioning为True
        x , c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        # print(type(x)) list 但应该是torch.tensor
        # x是隐层图片 c是文字编码后的向量
        # get_input的k参数是类型名字
        feature = batch[self.control_key]
        if bs is not None:
            feature = feature[:bs]
        feature = feature.to(self.device)
        feature  = feature.to(memory_format=torch.contiguous_format).float()
        #这里的x是图片 要换成特征 shape要一致
        return x,dict(c_crossattn = [c],c_concat=[feature])
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        # print('x_noisy.shape',x_noisy.shape)
        # print('t.shape',t.shape)
        # print('cond["feature"].shape',cond["feature"].shape)
        # exit()
        #此时的condition是feature
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        #如果有 get_input也要改
        # 我还是觉得需要cond_txt来训练
        # assert 'c_crossattn' in cond.keys()
        # print(cond["c_concat"].shape)
        cond_txt = torch.cat(cond['c_crossattn'] , 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None)
        else:
            feature = cond['c_concat'][0]
            control = self.control_model(x=feature, timesteps=t,context = cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        #是用于数据集batch的处理的 N应该是batch里面的数据个数
        use_ddim = ddim_steps is not None
        log = dict()
        #这个c是condition字典 这个函数是用上面实现的那个
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_feature, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        # 在干啥 归一化？
        # log["control"] = c_cat * 2.0 - 1.0
        log["control"] = c_feature
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_feature],"c_crossattn":[c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_feature = c_feature  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_feature],"c_crossattn":[uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_feature],"c_crossattn":[c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        #因为没有图片参考了所以只能赋值
        h, w = 512,512
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
