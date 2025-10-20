import torch



def cal_s_fp(x, Qmax, epsilon):
    dim = [-1]
    xmax = x.abs().amax(dim=dim, keepdim=True)  # shape: broadcastable to x along last dim
    mask = (xmax == 0)
    xmax_safe = xmax + epsilon * mask
    s = xmax_safe / Qmax
    return s



def fp_quant(x, bit, e_bit=4, m_bit=3, dim=-1, group_size=-1, e8_scale=False,e8_scale_op='ceil', scale_quant=False, scale_quant_2=False, epsilon=1e-25):
    if bit >= 16:
        return x
    assert bit == e_bit + m_bit + 1
    # assert e_bit > 0 and m_bit > 0, "e_bit and m_bit must be positive"
    bias = 2 ** (e_bit - 1) - 1  if e_bit > 0 else 1
    if e_bit > 0:
        Elow = -bias  # minimum exponent (after considering bias)
        Ehigh = 2 ** (e_bit) - 1 - bias  # maximum exponent (after considering bias)
    else:
        Elow = 0
        Ehigh = 0
    Mhigh = 2 ** m_bit - 1  # maximum mantissa
    if e_bit > 0:
        if e_bit == 4 and m_bit == 3:
            Qmax = 448
        elif e_bit ==5 and m_bit ==2 :
            Qmax = 57344
        else:
            Qmax = (1 + Mhigh / (Mhigh + 1)) * (2 ** Ehigh)
    else:
        Qmax = (0 + Mhigh / (Mhigh + 1)) * (2 ** Ehigh)  # maximum representable value
    Qmin = -Qmax  # minimum representable value
    # Handle dim: transpose to make dim the last dimension
    if dim != -1 and dim != len(x.shape) - 1:
        perm = list(range(len(x.shape)))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = x.permute(*perm)
    original_shape = x.shape
    if group_size > 0:
        new_shape = x.shape[:-1] + (-1, group_size)
        x = x.reshape(new_shape)
    
    s = cal_s_fp(x, Qmax, epsilon).to(x.dtype)
    assert (e8_scale and scale_quant) is False
    if e8_scale:
        if e8_scale_op == 'ceil':
            s = (2**(s.log2().ceil().clamp(-127, 127))).to(dtype=x.dtype)
        elif e8_scale_op == 'floor':
            s = (2**(s.log2().floor().clamp(-127, 127))).to(dtype=x.dtype)
        elif e8_scale_op == 'round':
            s = (2**(s.log2().round().clamp(-127, 127))).to(dtype=x.dtype)
        elif e8_scale_op == 'ocp':
            e8m0 = x.abs().amax(dim=dim, keepdim=True).log2().floor()-torch.tensor(Qmax, dtype=torch.float32, device=x.device).log2().floor()
            s = (2**(e8m0.clamp(-127, 127))).to(dtype=x.dtype)
        else:
            raise ValueError(f"e8_scale_op {e8_scale_op} not supported")
    if scale_quant:
        # E4M3 quant for scale + bf16 per-tensor scale, follow NVFP4
        s_of_s = s.abs().max()/ 448
        quant_s = (s/s_of_s).to(torch.float8_e4m3fn)
        s = s_of_s * quant_s.bfloat16()
    if scale_quant_2:
        # E4M3 quant for scale + E8M0 per-tensor scale
        s_of_s = s.abs().max()/ 448
        s_of_s = s_of_s.clamp(1e-25, 1e25) 
        s_of_s = (2**(s_of_s.log2().ceil().clamp(-127, 127))).to(dtype=x.dtype)
        s_of_s = s_of_s.clamp(1e-25, 1e25) 
        quant_s = (s/s_of_s).clamp(max=448).to(torch.float8_e4m3fn)
        s = s_of_s * quant_s.bfloat16()
    s = s.clamp(1e-25, 1e25) 
    x = x / s
    sign, x_abs = x.sign(), x.abs()
    expo = torch.floor(torch.log2(x.abs() + epsilon))
    expo = torch.clamp(expo, min=Elow, max=Ehigh)

    is_subnormal = expo <= Elow
    # normalized number
    mantissa_norm = x_abs / (2 ** expo) - 1  # in [0, 1)
    scale_m = 2 ** m_bit
    m_frac_int = torch.round(mantissa_norm * scale_m)  # in {0, 1, ..., 2^m}
    carry = (m_frac_int >= scale_m)  # == 2^m 
    m_frac_int = torch.where(carry, torch.zeros_like(m_frac_int), m_frac_int)
    mantissa_norm_q = m_frac_int / scale_m
    expo_adj = expo + carry.to(expo.dtype)
    # ============================================================
    # subnormalized number
    expo_sub = 1 - bias
    mantissa_sub = x_abs / (2 ** expo_sub)  # in [0, 1)
    m_sub_int = torch.round(mantissa_sub * scale_m)  # in {0, ..., 2^m}
    mantissa_sub_q = m_sub_int / scale_m
    # compose
    y = torch.where(
        is_subnormal,
        sign * (2 ** expo_sub) * mantissa_sub_q,
        sign * (2 ** expo_adj) * (1 + mantissa_norm_q)
    )
    y = y.clamp(Qmin, Qmax) * s


    if group_size > 0:
        y = y.reshape(original_shape)
    
    # Transpose back to original shape if dim was changed
    if dim != -1 and dim != len(original_shape) - 1:
        perm_back = list(range(len(original_shape)))
        perm_back[dim], perm_back[-1] = perm_back[-1], perm_back[dim]
        y = y.permute(*perm_back)
    return y

def print_fp_quant(e_bit=5, m_bit=2):
    assert e_bit > 0 and m_bit > 0, "e_bit and m_bit must be positive"
    bias = 2 ** (e_bit - 1) - 1  # bias value, used to map the exponent to a non-negative range
    if e_bit > 0:
        Elow = -bias  # minimum exponent (after considering bias)
        Ehigh = 2 ** (e_bit) - 1  - bias  # maximum exponent (after considering bias)
    else:
        Elow = 0
        Ehigh = 0
    Mhigh = 2 ** m_bit - 1  # maximum mantissa
    Qmax = (1 + Mhigh / (Mhigh + 1)) * (2 ** Ehigh)  # maximum representable value
    Qmin = -Qmax  # minimum representable value
    quant_data = fp_quant(torch.randn(1,4096), e_bit, m_bit)
    print(f"FP (E{e_bit}M{m_bit})Quant: Elow: {Elow}, Ehigh: {Ehigh}, Mhigh: {Mhigh}, Qmax: {Qmax}, Qmin: {Qmin}")
    print(f"unique num in  4096 number: {len(torch.unique(quant_data))}")


def int_quant(x, bit, dim=-1, group_size=-1, e8_scale=False, e8_scale_op="ceil", clip_style="sym", scale_quant=False, scale_quant_2=False):
    if bit >= 16:
        return x
    qmax = 2**(bit - 1) - 1
    if clip_style == "sym":
        qmin = -qmax
    elif clip_style == "asym":
        qmin = -2**(bit - 1)
    else:
        raise ValueError("clip_style must be 'sym' or 'asym'")
    if group_size < 0:
        group_size = x.shape[dim]
    # Reshape tensor to group elements
    shape = x.shape
    dim_size = shape[dim]
    if dim == -1:
        dim = len(shape)-1
    if dim_size % group_size != 0:
        raise ValueError("Dimension size must be divisible by group_size")
    num_groups = dim_size // group_size
    # Reshape to (..., num_groups, group_size, ...)
    new_shape = shape[:dim] + (num_groups, group_size) + shape[dim+1:]
    x_grouped = x.reshape(new_shape)
    # Update dim for grouped quantization
    group_dim = dim + 1 if dim >= 0 else len(new_shape) + dim
    xmax = x_grouped.abs().amax(dim=group_dim, keepdim=True)
    # quant_range = 2 * xmax
    # s = quant_range / (2**bit - 1)
    s = xmax/qmax
    assert (e8_scale and scale_quant) is False
    if e8_scale:
        if e8_scale_op == "ceil":
            s = (2**(s.log2().ceil().clamp(-127, 127))).to(dtype=x.dtype)
        elif e8_scale_op == "floor":
            s = (2**(s.log2().floor().clamp(-127, 127))).to(dtype=x.dtype)
        elif e8_scale_op == "round":
            s = (2**(s.log2().round().clamp(-127, 127))).to(dtype=x.dtype)
        elif e8_scale_op == "ocp":
            s = (2**((xmax.log2().floor()-torch.tensor(qmax, dtype=torch.float32, device=x.device).log2().floor()).clamp(-127, 127))).to(dtype=x.dtype)
        else:
            raise ValueError("e8_scale_op must be 'ceil', 'floor', 'round' or 'ocp'")
    if scale_quant:
        # E4M3 quant for scale, follow NVFP4
        s_of_s = s.abs().max().float()/ 448
        quant_s = (s/s_of_s).to(torch.float8_e4m3fn)
        s = s_of_s * quant_s.bfloat16()
    if scale_quant_2:
        # E4M3 quant for scale + E8M0 per-tensor scale
        s_of_s = s.abs().max()/ 448
        s_of_s = s_of_s.clamp(1e-25, 1e25)
        s_of_s = (2**(s_of_s.log2().ceil().clamp(-127, 127))).to(dtype=x.dtype)
        quant_s = (s/s_of_s).to(torch.float8_e4m3fn)
        s = s_of_s * quant_s.bfloat16()
    s = s.clamp(1e-25, 1e25)  # avoid zero
    x_scaled = x_grouped / s
    if torch.isnan(x_scaled).any():
        print("x_scaled is nan")
        import pdb;pdb.set_trace()
    x_rounded = x_scaled.round()
    x_quant = (x_rounded.clamp(qmin, qmax) * s).reshape(shape)
    return x_quant