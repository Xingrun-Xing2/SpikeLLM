import torch
import torch.nn as nn
import torch.nn.functional as F
from spike_driven_quant.quantizer import UniformAffineQuantizer


class SpikeQuantMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict = {},
        x2_quant_params: dict = {},
        disable_act_quant=False,
        matmul_func=torch.bmm,
        is_p = False,
        width = 0,
    ):
        super().__init__()
        # de-activate the quantized forward default
        self.use_act_quant = False
        # initialize quantizer
        self.i_cluster_counts = None
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)

        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        self.x2_quantizer_high = UniformAffineQuantizer(**x2_quant_params, high=True, Refine=True)

        self.matmul_func = matmul_func

        self.disable_act_quant = disable_act_quant

        self.low_p = x2_quant_params["low_p"]
        self.sum = None
        self.nsamples = 0
        
        mask_low_mse = torch.ones(width, dtype=torch.bool)
        self.register_buffer('mask_low_mse', mask_low_mse)
        
        self.is_p = is_p
        self.width = width
        self.mode = "fake_binary_simulate"

    @torch.no_grad()
    def add_batch(self, inp, out):        
        qp = (inp[0].data).to(dtype=torch.float64) # q [b h l c_h] # attn [b h l l]
        kv = (inp[1].data).to(dtype=torch.float64) # k [b h c_h l] # v [b h l c_h]

        grad = torch.matmul(qp.transpose(2,3), qp) # [b h c_h c_h] # [b h l l]
        grad = torch.matmul(grad, kv)              # [b h c_h l]   # [b h l c_h]
        inp = grad * kv

        self.nsamples += 1
        if self.sum==None:
            self.sum = inp
        else:
            self.sum += inp

    @torch.no_grad()
    def static(self):
        self.sum /= self.nsamples
        _, self.mask_low_mse = self.get_channel_mask(self.sum)
        del self.sum

    @torch.no_grad()
    def get_channel_mask(self, input):
        if self.is_p:
            mean = input.mean(2).unsqueeze(2)
        else:
            mean = input.mean(3).unsqueeze(3)
        
        low_quota = int(self.low_p * self.width)
        thresh = torch.sort(mean.flatten())[0][low_quota - 1]
        mask_low = mean <= thresh

        return mean, mask_low

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2):
        mask_low = self.mask_low_mse
        if self.use_act_quant:
            '''The equivalence of quantization-SNN conversion is addressed in Appendix A.3. Binary simulate is also proved equal to quantization 
            recently by Spike-Driven Transformer-V3: Yao M, Qiu X, Hu T, et al. Scaling spike-driven transformer with efficient spike firing approximation training[J]. 
            IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.'''
            if self.training or self.mode == "fake_binary_simulate" or self.mode == "fake_quant":
                x2 = self.x2_quantizer(x2) * mask_low + self.x2_quantizer_high(x2) * (~mask_low)
            elif self.mode == "multibit_simulate":
                self.x2_quantizer_high.per_token_dynamic_calibration(x2)
                inputs_int, scale, round_zero_point = self.x2_quantizer_high.fake_quant_int(x2, self.x2_quantizer_high.scale, self.x2_quantizer_high.round_zero_point)
                inputs_int = inputs_int * (~mask_low)
                spikes = []
                mem = torch.zeros_like(inputs_int)
                for i in range(self.steps):
                    if i == 0:
                        mem = inputs_int
                    else:
                        mem = mem - spikes[i - 1]
                    
                    spike = torch.where(mem>=self.step_levels, self.step_levels-1, mem).to(mem.dtype).to(mem.device)
                    spikes.append(spike)
                
                spikes = torch.stack(spikes,dim=0)

                x_dequant = self.x2_quantizer_high.fake_dequant_int(spikes.sum(0), self.x2_quantizer_high.scale, self.x2_quantizer_high.round_zero_point)
                out = x_dequant * (~mask_low) + self.x2_quantizer(x2) * mask_low

                return out

        return x2

    def forward(self, double):
        x1, x2 = double
        
        x1 = self.quant_x1(x1)
        x2 = self.quant_x2(x2)

        out = self.matmul_func(x1, x2)
        return out
