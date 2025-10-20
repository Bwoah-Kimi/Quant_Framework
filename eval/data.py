from datasets import load_dataset
import torch
import random
from tqdm import tqdm
import torch.nn.functional as F

def get_wikitext2_test(tokenizer):
    print("get_wikitext2")
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    return testenc

def get_wikitext2_test_sep(tokenizer, n_samples=128,seqlen=2048):
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    testenc = testenc.input_ids
    testenc = testenc[:, :n_samples * seqlen]
    testenc = testenc.reshape(n_samples, seqlen)
    return testenc

# ADD THIS FUNCTION FOR C4
def get_c4_test(tokenizer, n_samples=128, seqlen=4096):
    """
    Loads and tokenizes the C4 validation set.
    """
    print("Loading and tokenizing C4 validation set...")
    # Using a streaming dataset to avoid downloading the whole thing
    dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
    
    tokenized_samples = []
    for sample in tqdm(dataset.take(n_samples), total=n_samples):
        tokenized_samples.append(tokenizer(sample['text'])['input_ids'])

    # Concatenate all tokenized samples into one long sequence
    all_tokens = [token for sample in tokenized_samples for token in sample]
    
    # Truncate to the desired total length
    all_tokens = all_tokens[:n_samples * seqlen]
    
    # Reshape into the final tensor
    test_data = torch.tensor(all_tokens).reshape(1, -1)
    print(f"C4 data loaded. Shape: {test_data.shape}")
    return test_data

# ADD THIS FUNCTION FOR PG-19
def get_pg19_test(tokenizer, n_samples=128, seqlen=4096):
    """
    Loads and tokenizes the PG-19 validation set.
    """
    print("Loading and tokenizing PG-19 validation set...")
    dataset = load_dataset('pg19', split='validation')
    
    # Take a subset for manageable evaluation
    text = "\n\n".join(dataset['text'][:n_samples])
    
    enc = tokenizer(text, return_tensors='pt')
    
    # Ensure it's not longer than n_samples * seqlen
    total_len = n_samples * seqlen
    if enc.input_ids.shape[1] > total_len:
        enc.input_ids = enc.input_ids[:, :total_len]
        enc.attention_mask = enc.attention_mask[:, :total_len]

    print(f"PG-19 data loaded. Shape: {enc.input_ids.shape}")
    return enc.input_ids

@torch.no_grad()
def cal_ppl(model, tokenizer, seqlen=4096, dataset='wikitext2', batch_size=1):
    if dataset == 'wikitext2':
        testloader = get_wikitext2_test(tokenizer)
        testenc = testloader.input_ids
    else:
        raise NotImplementedError

    nsamples = testenc.numel() // seqlen
    testenc = testenc[:, :nsamples * seqlen].view(nsamples, seqlen).contiguous()

    model.eval()
    nlls = []

    for i in tqdm(range(0, nsamples, batch_size), desc="Evaluating PPL"):
        j = min(i + batch_size, nsamples)
        
        batch = testenc[i:j, :]
        
        inputs = batch.to(model.device)
        labels = batch.to(model.device)
        
        outputs = model(inputs, labels=labels)

        neg_log_likelihood = outputs.loss * batch.size(0) * seqlen
        nlls.append(neg_log_likelihood)

    loss = torch.stack(nlls).sum()/ (nsamples * seqlen)
    ppl = torch.exp(loss)
    
    return ppl, loss

@torch.no_grad()
def cal_kl(quant_model, fp_model, tokenizer, seqlen=4096, dataset='wikitext2', topk=100):
    # Select the dataset based on the argument
    if dataset == 'wikitext2':
        testenc = get_wikitext2_test(tokenizer)
        testenc = testenc.input_ids
    elif dataset == 'c4':
        # Using a smaller number of samples for C4 because it's huge
        testenc = get_c4_test(tokenizer, n_samples=128, seqlen=seqlen)
    elif dataset == 'pg19':
        testenc = get_pg19_test(tokenizer, n_samples=32, seqlen=seqlen)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
    nsamples = testenc.numel() // seqlen
    print(f"Evaluating on {dataset} with {nsamples} samples...")
    quant_model.eval()
    fp_model.eval()
    kls = []
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)]
        # For C4/PG-19, the last batch might be smaller, skip it for simplicity
        if batch.shape[1] != seqlen:
            continue
        labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)]
        labels = labels.to(quant_model.device)
        inputs_quant = batch.to(quant_model.device)
        quant_logits = quant_model(inputs_quant, labels=inputs_quant).logits
        inputs_fp = batch.to(fp_model.device)
        fp_logits = fp_model(inputs_fp, labels=inputs_fp).logits
        fp_logits = fp_logits.to(quant_logits.device)
        fp_topk_logits, topk_ids = torch.topk(fp_logits, k=topk, dim=-1)
        quant_topk_logits = torch.gather(quant_logits, dim=-1, index=topk_ids,)
        # kl = F.kl_div(outputs1.logits.log_softmax(dim=-1), outputs2.logits.softmax(dim=-1), reduction='batchmean')
        kl = F.kl_div(quant_topk_logits.log_softmax(dim=-1), fp_topk_logits.softmax(dim=-1))
        # import pdb; pdb.set_trace()
        kls.append(kl*1e6)
    # import pdb; pdb.set_trace()
    avg_kl = torch.stack(kls).mean()
    return avg_kl
