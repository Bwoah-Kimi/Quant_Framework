import torch
from quant.quant_linear import QuantLinear
from tqdm import tqdm


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)

def wrap_to_quant_model(model, quant_param):
    '''
    replace nn.Linear and norm layer to correspond quantization counterparts
    '''
    for name, module in tqdm(model.named_modules(), desc="Quantizing Layers"):
        if isinstance(module,torch.nn.Linear):
            # skip lm_head quantization
            if 'lm_head' in name:
                continue
            quantlinear = QuantLinear.from_original_module(module, quant_param)
            set_op_by_name(model, name, quantlinear)  
            del module.weight
            del module  
            torch.cuda.empty_cache()

