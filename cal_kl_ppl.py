import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
import os
from quant.utils import wrap_to_quant_model
from tqdm import tqdm
from eval.data import get_wikitext2_test, get_c4_test, get_pg19_test
from accelerate import infer_auto_device_map, dispatch_model, init_empty_weights
from accelerate.hooks import remove_hook_from_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,default='./models/138M_100B', help='model path')
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--e8_scale", action='store_true', help="Enable e8 scale.")
    parser.add_argument("--scale_quant", action='store_true', help="Enable scale quant.")
    parser.add_argument("--scale_quant_2", action='store_true', help="Enable scale quant.")
    parser.add_argument("--w_quant_inplace", action='store_true', help="Enable inplace weight quant.")
    parser.add_argument("--debug", action='store_true', help="Enable inplace weight quant.")
    parser.add_argument("--e8_scale_op", type=str, default="ceil",choices=['ceil','floor','round','ocp'], help="")
    parser.add_argument("--clip_style", type=str, default="sym",choices=['sym','asym'], help="")
    parser.add_argument("--quant_type", type=str, default="int")
    parser.add_argument("--max_memory", type=str, default="73GB")
    parser.add_argument('--q1_w', type=float, default=16)
    parser.add_argument('--q1_x', type=float, default=16)
    parser.add_argument('--q2_w', type=float, default=16)
    parser.add_argument('--q2_g', type=float, default=16)
    parser.add_argument('--q3_x', type=float, default=16)
    parser.add_argument('--q3_g', type=float, default=16)
    parser.add_argument('--e_bit', type=float, default=5)
    parser.add_argument('--m_bit', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=-1)
    parser.add_argument('--rotate_dim', type=int, default=-1)
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='wikitext2', 
        choices=['wikitext2', 'c4', 'pg19'], 
        help='Dataset to use for KL divergence calculation.'
    )
    parser.add_argument('--topk', type=int, default=25)
    
    args = parser.parse_args()
    return args

@torch.no_grad()
def evaluate_ppl_and_kl(model_path, quant_args, tokenizer, seqlen=4096, dataset='wikitext2', topk=100):
    """
    The PPL and KL divergence of the model are evaluated in two stages to save memory.
    1. Load the FP model, compute and cache top-k logits to the CPU.
    2. Load Quant model and calculate PPL and KL divergence in one go.
    """
    device = 'cuda'

    # --- 数据集加载 ---
    if dataset == 'wikitext2':
        testenc = get_wikitext2_test(tokenizer).input_ids
    elif dataset == 'c4':
        testenc = get_c4_test(tokenizer, n_samples=128, seqlen=seqlen)
    elif dataset == 'pg19':
        testenc = get_pg19_test(tokenizer, n_samples=32, seqlen=seqlen)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
    
    nsamples = testenc.numel() // seqlen
    testenc = testenc[:, :nsamples * seqlen].view(nsamples, seqlen)
    print(f"Evaluating on {dataset} with {nsamples} samples...")

    # =================================================================
    # 阶Phase One: Cache the Top-K Logits of the FP model
    # =================================================================
    print("\n--- Stage 1: Caching FP model's top-k logits ---")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with init_empty_weights():
        fp_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    block_class_name = fp_model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(
        fp_model,
        max_memory={i: args.max_memory for i in range(torch.cuda.device_count())},
        no_split_module_classes=[block_class_name],
        verbose=False
    )
    fp_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map=device_map, dtype=torch.bfloat16
    )
    fp_model.eval()

    
    cached_fp_logits = []
    for i in tqdm(range(nsamples), desc="Caching FP Logits"):
        if i==2 and quant_args.debug:
            break
        batch = testenc[i:i+1, :]
        if batch.shape[1] != seqlen:
            continue
        
        inputs_fp = batch.to(device)
        fp_logits = fp_model(inputs_fp).logits
        
        fp_topk_logits, topk_ids = torch.topk(fp_logits, k=topk, dim=-1)
        
        cached_fp_logits.append((fp_topk_logits.to('cpu'), topk_ids.to('cpu')))
    
    torch.cuda.empty_cache()
    print("--- Stage 1 Complete: FP model unloaded. ---")

    # =================================================================
    # Phase Two: Load the Quant model and uniformly calculate PPL and KL
    # =================================================================
    print("\n--- Stage 2: Evaluating Quant model for PPL and KL ---")
    wrap_to_quant_model(fp_model, quant_args)
    quant_model = fp_model
    quant_model.eval()

    nlls = []
    kls = []
    
    for i, (fp_topk_logits_cpu, topk_ids_cpu) in tqdm(enumerate(cached_fp_logits), total=nsamples, desc="Evaluating Quant Model"):
        if i==2 and quant_args.debug:
            break
        batch = testenc[i:i+1, :]
        inputs_quant = batch.to(device)
        
        outputs = quant_model(inputs_quant, labels=inputs_quant)
        
        neg_log_likelihood = outputs.loss.item() * seqlen
        nlls.append(neg_log_likelihood)

        quant_logits = outputs.logits
        fp_topk_logits = fp_topk_logits_cpu.to(device)
        topk_ids = topk_ids_cpu.to(device)
        
        quant_topk_logits = torch.gather(quant_logits, dim=-1, index=topk_ids)
        
        kl_div = F.kl_div(quant_topk_logits.log_softmax(dim=-1),fp_topk_logits.softmax(dim=-1))
        kls.append(kl_div.item()*1e6)

    del quant_model
    torch.cuda.empty_cache()

    total_nll = sum(nlls)
    loss = total_nll / (nsamples * seqlen)
    ppl = torch.exp(torch.tensor(loss)).item()
    
    avg_kl = sum(kls) / len(kls) if kls else 0.0

    return ppl, loss, avg_kl


if __name__ == '__main__':
    args = parse_args()

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, add_bos_token=False)

    ppl, loss, kl = evaluate_ppl_and_kl(
        model_path=args.model_path,
        quant_args=args,
        tokenizer=tokenizer,
        dataset=args.dataset,
        topk=args.topk,
        seqlen=4096
    )

    print("\n" + "="*30)
    print("      Evaluation Results")
    print("="*30)
    print(f"Quant Model Perplexity (PPL): {ppl:.4f}")
    print(f"Quant Model Loss:             {loss:.4f}")
    print(f"Quant Model KL Divergence:    {kl:.4f}")
    print("="*30)