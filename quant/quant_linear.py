import random
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from quant.quant_func import int_quant,fp_quant
from quant.hadamard import random_hadamard_matrix, hadamard_rotate,generate_new_hadamard


class QuantParams:
    def __init__(self, config):
        self.q1_w = config.q1_w
        self.q1_x = config.q1_x
        self.q2_w = config.q2_w
        self.q2_g = config.q2_g
        self.q3_x = config.q3_x
        self.q3_g = config.q3_g
        self.group_size = config.group_size
        self.quant_type = config.quant_type
        self.e_bit = config.e_bit
        self.m_bit = config.m_bit
        self.e8_scale = config.e8_scale
        self.e8_scale_op = config.e8_scale_op
        self.rotate_dim = config.rotate_dim
        self.clip_style = config.clip_style
        self.scale_quant = config.scale_quant
        self.scale_quant_2 = config.scale_quant_2



class IntQuantLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, quant_params, rotate_h=None):
        # Forward quantization
        quantized_x = int_quant(x, quant_params.q1_x, group_size=quant_params.group_size,e8_scale=quant_params.e8_scale, e8_scale_op=quant_params.e8_scale_op, clip_style=quant_params.clip_style, scale_quant=quant_params.scale_quant, scale_quant_2=quant_params.scale_quant_2)
        quantized_weight = int_quant(weight, quant_params.q1_w, group_size=quant_params.group_size, e8_scale=quant_params.e8_scale, e8_scale_op=quant_params.e8_scale_op, clip_style=quant_params.clip_style, scale_quant=quant_params.scale_quant, scale_quant_2=quant_params.scale_quant_2)
        out = F.linear(quantized_x, quantized_weight, bias)
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias, quantized_weight, quantized_x, rotate_h)
        ctx.quant_params = quant_params
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, quantized_weight, quantized_x, rotate_h = ctx.saved_tensors
        quant_params = ctx.quant_params
        
        quantized_grad_output =  int_quant(grad_output, quant_params.q2_g, group_size=quant_params.group_size, e8_scale=quant_params.e8_scale, e8_scale_op=quant_params.e8_scale_op, clip_style=quant_params.clip_style, scale_quant=quant_params.scale_quant, scale_quant_2=quant_params.scale_quant_2)
        quantized_weight_t = int_quant(weight.t(), quant_params.q2_w, group_size=quant_params.group_size, e8_scale=quant_params.e8_scale, e8_scale_op=quant_params.e8_scale_op, clip_style=quant_params.clip_style, scale_quant=quant_params.scale_quant, scale_quant_2=quant_params.scale_quant_2)
        grad_input = F.linear(quantized_grad_output, quantized_weight_t)
        quantized_grad_output_t =  int_quant(grad_output.transpose(1,2), quant_params.q3_g, e8_scale=quant_params.e8_scale, e8_scale_op=quant_params.e8_scale_op, clip_style=quant_params.clip_style, scale_quant=quant_params.scale_quant, scale_quant_2=quant_params.scale_quant_2)
        quantized_x =  int_quant(x, bits=quant_params.q3_x, dim=1, group_size=quant_params.group_size, e8_scale=quant_params.e8_scale, e8_scale_op=quant_params.e8_scale_op, clip_style=quant_params.clip_style, scale_quant=quant_params.scale_quant, scale_quant_2=quant_params.scale_quant_2)
        grad_weight = torch.matmul(quantized_grad_output_t, quantized_x)
        grad_weight = grad_weight.sum(dim=0)
        grad_bias = quantized_grad_output.sum(0) if bias is not None else None
        
        return grad_input, grad_weight, grad_bias, None, None, None

class FPQuantLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, quant_params, rotate_h=None):
        # Forward quantization
        quantized_x = fp_quant(x, quant_params.q1_x, quant_params.e_bit, quant_params.m_bit, group_size=quant_params.group_size, e8_scale=quant_params.e8_scale, e8_scale_op=quant_params.e8_scale_op, scale_quant=quant_params.scale_quant, scale_quant_2=quant_params.scale_quant_2)
        quantized_weight = fp_quant(weight, quant_params.q1_w, quant_params.e_bit, quant_params.m_bit, group_size=quant_params.group_size, e8_scale=quant_params.e8_scale, e8_scale_op=quant_params.e8_scale_op, scale_quant=quant_params.scale_quant, scale_quant_2=quant_params.scale_quant_2)
        out = F.linear(quantized_x, quantized_weight, bias)
        
        # Save for backward pass
        ctx.save_for_backward(x, weight, bias, quantized_x, rotate_h)
        ctx.quant_params = quant_params
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, quantized_x, rotate_h = ctx.saved_tensors
        quant_params = ctx.quant_params
        
        # Backward quantization
        # Calculate grad_input: quantized_grad_output @ quantized_weight
        # if rotate_h is not None:
        #     quantized_grad_output =  fp_quant(hadamard_rotate(grad_output,rotate_h), quant_params.q2_g, quant_params.e_bit, quant_params.m_bit, stochastic=quant_params.stochastic, group_size=quant_params.group_size)
        #     quantized_weight_t = fp_quant(hadamard_rotate(weight.t(),rotate_h), quant_params.q2_w, quant_params.e_bit, quant_params.m_bit, group_size=quant_params.group_size)
        # else:
        quantized_grad_output =  fp_quant(grad_output, quant_params.q2_g, quant_params.e_bit, quant_params.m_bit,group_size=quant_params.group_size, e8_scale=quant_params.e8_scale, e8_scale_op=quant_params.e8_scale_op, scale_quant=quant_params.scale_quant, scale_quant_2=quant_params.scale_quant_2)
        quantized_weight_t = fp_quant(weight.t(), quant_params.q2_w, quant_params.e_bit, quant_params.m_bit, group_size=quant_params.group_size, e8_scale=quant_params.e8_scale, e8_scale_op=quant_params.e8_scale_op, scale_quant=quant_params.scale_quant, scale_quant_2=quant_params.scale_quant_2)
        grad_input = F.linear(quantized_grad_output, quantized_weight_t)
        # Calculate grad_weight: quantized_grad_output^T @ quantized_x
        # if rotate_h is not None:
        #     quantized_grad_output_t =  fp_quant(hadamard_rotate(grad_output.transpose(1,2),rotate_h), quant_params.q3_g, quant_params.e_bit, quant_params.m_bit, stochastic=quant_params.stochastic, group_size=quant_params.group_size)
        #     quantized_x =  fp_quant(hadamard_rotate(x.transpose(1,2),rotate_h).transpose(1,2), quant_params.q3_x, quant_params.e_bit, quant_params.m_bit, dim=[1], group_size=quant_params.group_size)
        # else:
        quantized_grad_output_t =  fp_quant(grad_output.transpose(1,2), quant_params.q3_g, quant_params.e_bit, quant_params.m_bit, group_size=quant_params.group_size, e8_scale=quant_params.e8_scale, e8_scale_op=quant_params.e8_scale_op, scale_quant=quant_params.scale_quant, scale_quant_2=quant_params.scale_quant_2)
        quantized_x =  fp_quant(x, quant_params.q3_x, quant_params.e_bit, quant_params.m_bit, dim=1, group_size=quant_params.group_size, e8_scale=quant_params.e8_scale, e8_scale_op=quant_params.e8_scale_op, scale_quant=quant_params.scale_quant, scale_quant_2=quant_params.scale_quant_2)
        grad_weight = torch.matmul(quantized_grad_output_t, quantized_x)
        grad_weight = grad_weight.sum(dim=0)

        # If bias exists, calculate grad_bias
        grad_bias = quantized_grad_output.sum(0) if bias is not None else None
        
        return grad_input, grad_weight, grad_bias, None, None, None



class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        config,
        bias=False,
        dtype=torch.float32,
        device='cuda'
    ):
        super(QuantLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).to(dtype=dtype, device=device))
        else:
            self.register_parameter('bias', None)
        
        # Get quantization parameters from config
        self.quant_params = QuantParams(config)
        if self.quant_params.rotate_dim > 0:
            # self.rotate_h =  random_hadamard_matrix(self.quant_params.rotate_dim,device).to(dtype=dtype)
            rotate_h =  random_hadamard_matrix(self.quant_params.rotate_dim,device).to(dtype=dtype)
            new_rotate_h = generate_new_hadamard(rotate_h)
            self.rotate_h = new_rotate_h
        else:
            self.rotate_h = None

    def __repr__(self) -> str:
            if self.quant_params.quant_type == "fp":
                return (
                    f"{self.__class__.__name__}("
                    f"in_features={self.in_features}, "
                    f"out_features={self.out_features}, "
                    f"bias={self.bias is not None},"
                    f"quant_type={self.quant_params.quant_type}, "
                    f"x_bit={self.quant_params.q1_x}, "
                    f"w_bit={self.quant_params.q1_w}, "
                    f"e_bit={self.quant_params.e_bit}, "
                    f"m_bit={self.quant_params.m_bit}, "
                    f"group_size={self.quant_params.group_size}, "
                    f"e8_scale={self.quant_params.e8_scale}, "
                    f"rotate_dim={self.quant_params.rotate_dim})"
                )
            else:
                return (
                    f"{self.__class__.__name__}("
                    f"in_features={self.in_features}, "
                    f"out_features={self.out_features}, "
                    f"bias={self.bias is not None},"
                    f"quant_type={self.quant_params.quant_type}, "
                    f"x_bit={self.quant_params.q1_x}, "
                    f"w_bit={self.quant_params.q1_w}, "
                    f"group_size={self.quant_params.group_size}, "
                    f"e8_scale={self.quant_params.e8_scale}, "
                    f"rotate_dim={self.quant_params.rotate_dim})"
                )
    @classmethod
    def from_original_module(cls, original_module: nn.Linear, config):
        """     
        Args:
            cls: The class itself (QuantLinear).
            original_module (nn.Linear): The original linear layer to copy weights from.
            config: The configuration object for quantization.
        """
        in_features = original_module.in_features
        out_features = original_module.out_features
        has_bias = original_module.bias is not None
        new_module = cls(
            in_features=in_features,
            out_features=out_features,
            config=config,
            bias=has_bias,
            dtype=original_module.weight.dtype,
            device=original_module.weight.device
        )

        with torch.no_grad():
            if config.w_quant_inplace:
                ori_weight = original_module.weight
                quant_weight = int_quant(ori_weight, config.q1_w, group_size=config.group_size, e8_scale=config.e8_scale, e8_scale_op=config.e8_scale_op, clip_style=config.clip_style, scale_quant=config.scale_quant)
                new_module.weight.copy_(quant_weight)
                new_module.quant_params.q1_w = 16
            else:
                new_module.weight.copy_(original_module.weight.clone().detach())
            if has_bias:
                new_module.bias.copy_(original_module.bias.clone().detach())
            if new_module.quant_params.rotate_dim > 0:
                new_module.rotate_h = new_module.rotate_h.to(new_module.weight.device, dtype=new_module.weight.dtype)
        
        return new_module
        
    def forward(self, x):
        if self.quant_params.rotate_dim > 0:
            x = hadamard_rotate(x, self.rotate_h)
            weight = hadamard_rotate(self.weight, self.rotate_h)
        else:
            weight = self.weight
        if self.quant_params.quant_type == "fp":
            return FPQuantLinearFunction.apply(x, weight, self.bias, self.quant_params, self.rotate_h)
        elif self.quant_params.quant_type == "int":
            return IntQuantLinearFunction.apply(x, weight, self.bias, self.quant_params, self.rotate_h)
        else:
            raise ValueError(f"Unknown quant_type: '{self.quant_params.quant_type}'")