from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
import os
from eval.data import get_wikitext2_test_sep
import torch
import torch.nn as nn
from quant.quant_func import fp_quant, int_quant
import pandas as pd # NEW: Import pandas for writing to Excel
from quant.hadamard import random_hadamard_matrix, hadamard_rotate



@torch.no_grad()
def cal_quantization_error(data, quant_dim, quant_type, group_size, bit=None, 
e_bit=None, m_bit=None,  e8_scale=False, e8_scale_op=None, clip_style=None,
scale_quant=False, scale_quant_2=False, metric_type="qsnr", rotate_dim=-1):
    if rotate_dim > 0:
        hadamard_matrix = random_hadamard_matrix(rotate_dim, device=data.device).to(data.dtype)
        data = hadamard_rotate(data, hadamard_matrix,quant_dim)
    if quant_type == "fp":
        quant_data = fp_quant(data, bit, e_bit, m_bit, quant_dim, group_size, e8_scale, e8_scale_op, scale_quant, scale_quant_2)
    elif quant_type == "int":
        quant_data = int_quant(data, bit, quant_dim, group_size, e8_scale, e8_scale_op, clip_style, scale_quant, scale_quant_2)
    else:
        raise ValueError("quant_type must be fp or int")
    if metric_type == "mre":
        mre = ((quant_data - data).abs()/(data.abs()+1e-15)).mean()*100
        return mre
    elif metric_type == "qsnr":
        qsnr = -10 * torch.log10(torch.mean((quant_data.float() - data.float())**2) / (torch.mean(data.float()**2)))
        return qsnr
    elif metric_type == "underflow":
        underflow = (quant_data == 0).float().mean()*100 -  (data == 0).float().mean()*100
        return underflow
    else:
        raise NotImplementedError

def get_activation_hook(layer_name):
    def hook(model, input, output):
        if isinstance(input, tuple):
            input = input[0].detach()
        if isinstance(output, tuple):
            output = output[0].detach()
        input_activation[layer_name] = input
        output_activation[layer_name] = output
    return hook

def get_gradient_hook(layer_name):
    def hook(module, grad_output, grad_input):
        gradients[layer_name] = {
            "dy": grad_input[0].detach(),
            'dx': grad_output[0].detach(),
        }
    return hook


class LayerWiseData:
    def __init__(self):
        self.q_proj = []
        self.k_proj = []
        self.v_proj = []
        self.o_proj = []
        self.up_proj = []
        self.gate_proj = []
        self.down_proj= []
    
    def append(self, data, name):
        if "q_proj" in name:
            self.q_proj.append(data.item())
        elif "k_proj" in name:
            self.k_proj.append(data.item())
        elif "v_proj" in name:
            self.v_proj.append(data.item())
        elif "o_proj" in name:
            self.o_proj.append(data.item())
        elif "up_proj" in name:
            self.up_proj.append(data.item())
        elif "gate_proj" in name:
            self.gate_proj.append(data.item())
        elif "down_proj" in name:
            self.down_proj.append(data.item())
        else:
            raise ValueError("name must be q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj or down_proj")
        
    def clear(self):
        self.q_proj = []
        self.k_proj = []
        self.v_proj = []
        self.o_proj = []
        self.up_proj = []
        self.gate_proj = []
        self.down_proj= []

    def get_mean(self, name):
        if "q_proj" in name:
            return torch.tensor(self.q_proj).mean(dim=0)
        elif "k_proj" in name:
            return torch.tensor(self.k_proj).mean(dim=0)
        elif "v_proj" in name:
            return torch.tensor(self.v_proj).mean(dim=0)
        elif "o_proj" in name:
            return torch.tensor(self.o_proj).mean(dim=0)
        elif "up_proj" in name:
            return torch.tensor(self.up_proj).mean(dim=0)
        elif "gate_proj" in name:
            return torch.tensor(self.gate_proj).mean(dim=0)
        elif "down_proj" in name:
            return torch.tensor(self.down_proj).mean(dim=0)
        else:
            raise ValueError("name must be q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj or down_proj")


    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,default='./models/138M_100B', help='model path')
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--e8_scale", action='store_true', help="Enable e8 scale.")
    parser.add_argument("--e8_scale_op", type=str, default="ceil",choices=['ceil','floor','round'], help="")
    parser.add_argument("--scale_quant", action='store_true', help="Enable scale quant.")
    parser.add_argument("--scale_quant_2", action='store_true', help="Enable scale quant.")
    parser.add_argument("--clip_style", type=str, default="sym",choices=['sym','asym'], help="")
    parser.add_argument("--quant_type", type=str, default="int")
    parser.add_argument("--metric_type", type=str, default="qsnr")
    parser.add_argument('--bit', type=int, default=16)
    parser.add_argument('--e_bit', type=float, default=4)
    parser.add_argument('--m_bit', type=int, default=3)
    parser.add_argument('--group_size', type=int, default=-1)
    parser.add_argument('--rotate_dim', type=int, default=-1)
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='wikitext2', 
        choices=['wikitext2', 'c4', 'pg19'], 
        help='Dataset to use for KL divergence calculation.'
    )
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--topk', type=int, default=25)
    parser.add_argument('--verbose', action='store_true', help="Enable verbose mode.")
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output Excel files.')
    parser.add_argument('--filename_prefix', type=str, default='', help='Prefix for the output Excel filenames.')
    
    args = parser.parse_args()
    return args

args = parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True,add_bos_token=False)
config = AutoConfig.from_pretrained(args.model_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path,trust_remote_code=True,device_map='auto',dtype=torch.bfloat16)

input_activation = {}
output_activation = {} # not used now
gradients = {}
for name, layer in model.named_modules():
    if isinstance(layer, (torch.nn.Linear)):
        layer.register_forward_hook(get_activation_hook(name))
        layer.register_backward_hook(get_gradient_hook(name))

datas = get_wikitext2_test_sep(tokenizer, n_samples=args.n_samples, seqlen=args.seqlen+1)
datas = datas.to(model.device)
inputs = datas[:, :-1]  # [batch_size, seq_len-1]
targets = datas[:, 1:]  # [batch_size, seq_len-1]

criterion = nn.CrossEntropyLoss()
model.train()
outputs = model(inputs)[0]
# Reshape for loss calculation
outputs = outputs.reshape(-1, outputs.size(-1))  # [batch_size * (seq_len-1), vocab_size]
targets = targets.reshape(-1)  # [batch_size * (seq_len-1)]
# Calculate loss
loss = criterion(outputs, targets)
loss.backward()
data = LayerWiseData()
detailed_data = [] 
summary_data = [] # NEW: List to hold the summary (average over all layers) results
group_size_list = [-1, 256, 128, 64, 32, 16]
layer_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj']
sections = [
    ("Input Activation (per-row)", input_activation.items(), lambda x: x, -1),
    ("Weight (per-column)", model.named_modules(), lambda layer: layer.weight, -1),  # weight will transpose during computation
    ("dy (per-row)", gradients.items(), lambda x: x['dy'], -1),
    ("Weight (per-row)", model.named_modules(), lambda layer: layer.weight, 0),
    ("Input Activation (per-column)", input_activation.items(), lambda x: x, 1),
    ("dy (per-column)", gradients.items(), lambda x: x['dy'], 1),
]
# Process each section and group size
for section_name, iterator, get_data, quant_dim in sections:
    print(f"---------------{section_name}-----------------")
    error_metric_results = {gs: LayerWiseData() for gs in group_size_list}
    
    for layer_name, item in iterator:
        if "Weight" in section_name:
            if "named_modules" in str(iterator) and not isinstance(item, (torch.nn.Linear)):
                continue
        if "head" in layer_name:
            continue
        data_tensor = get_data(item)
        for gs in group_size_list:
            error_metric = cal_quantization_error(data_tensor, quant_dim, args.quant_type, 
            gs, args.bit, args.e_bit, args.m_bit, metric_type=args.metric_type, 
            e8_scale=args.e8_scale, e8_scale_op=args.e8_scale_op, clip_style=args.clip_style, 
            scale_quant=args.scale_quant, scale_quant_2=args.scale_quant_2,
            rotate_dim=args.rotate_dim)
            if args.verbose:
                print(f"layer_name: {layer_name}, shape: {data_tensor.shape}, group_size: {gs}, error_metric: {error_metric.item():.5f}")
            error_metric_results[gs].append(error_metric, layer_name)
    
    # NEW: A temporary list to hold the numeric results for the current section to make averaging easier
    current_section_metrics = []

    # Collect means for each layer type and group size 
    for layer_type in layer_types: 
        row = [section_name, layer_type] 
        for gs in group_size_list: 
            error_metric_value = error_metric_results[gs].get_mean(layer_type).item() 
            row.append(error_metric_value) 
        detailed_data.append(row)
        current_section_metrics.append(row[2:]) # NEW: Add only the numeric values for averaging

    # NEW: Calculate the average across all layer types for the current section
    if current_section_metrics:
        metrics_tensor = torch.tensor(current_section_metrics)
        avg_metrics = metrics_tensor.mean(dim=0).tolist() # Average across layer types
        summary_row = [section_name] + avg_metrics
        summary_data.append(summary_row)
    # Clear data for each group size
    for gs in group_size_list:
        error_metric_results[gs].clear()

# --- NEW: Save results to Excel files ---
# Ensure the output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Create the filename prefix (add an underscore if a prefix is provided)
prefix = f"{args.filename_prefix}_" if args.filename_prefix else ""

# Define the full file paths
detailed_filename = f"{prefix}detailed_{args.metric_type}_results.xlsx"
summary_filename = f"{prefix}summary_{args.metric_type}_results.xlsx"
detailed_filepath = os.path.join(args.output_dir, detailed_filename)
summary_filepath = os.path.join(args.output_dir, summary_filename)

# 1. Save the detailed per-layer-type results
detailed_headers = ['Section', 'Layer Type'] + [f'GS={gs}' for gs in group_size_list]
df_detailed = pd.DataFrame(detailed_data, columns=detailed_headers)
df_detailed.to_excel(detailed_filepath, index=False, float_format="%.2f")
print(f"\n✅ Detailed per-layer QSNR results saved to '{detailed_filename}'")

# 2. Save the summary (model-wide average) results
summary_headers = ['Section'] + [f'GS={gs}' for gs in group_size_list]
df_summary = pd.DataFrame(summary_data, columns=summary_headers)
df_summary.to_excel(summary_filepath, index=False, float_format="%.2f")
print(f"✅ Model-wide average QSNR results saved to '{summary_filename}'")