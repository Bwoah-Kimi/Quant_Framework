# INT v.s. FP
We demonstrate that MXINT8/NVINT4 quantization formats are more accurate than their FP counterparts form both theoretical and practical scenarios. More details can be found in our paper [INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats](**). 

# Introduction
- Theoretical QSNR
![theoretical](./asserts/theoretical_qsnr.png)
- Tensor-wise QSNR
![tensor_qsnr](./asserts/tensor_wise_qsnr.png)



# Content of this code repo
- `cal_qsnr.py`: calculate the QSNR of quantized models.
- `cal_kl_ppl.py`: calculate the KL divergence and perplexity of quantized models.
- `quant/quant_func.py`: fake quantization function of integer and float-point quantization
- `quant/quant_linear.py`: linear layer with integer and float-point quantization

# Evaluate KL divergence and perplexity of quantized models
We offer the scripts to evaluate the KL divergence and perplexity of quantized models across different low-bit formats. You can change `--model_path` to evaluate the KL divergence and perplexity of other models. You can also set `--rotate_dim 32` to introduce 32x32 random Hadamard rotation before quantization.

- MXINT8
```
CUDA_VISIBLE_DEVICES=0 python3 cal_kl_ppl.py --model_path ./hf_models/Llama-3.2-1B --q1_x 8 --q1_w 8 --group_size 32 --e8_scale --e8_scale_op ceil
```
- MXFP8
```
CUDA_VISIBLE_DEVICES=0 python3 cal_kl_ppl.py --model_path ./hf_models/Llama-3.2-1B --quant_type fp --e_bit 4 --m_bit 3 --q1_x 8 --q1_w 8 --group_size 32 --e8_scale --e8_scale_op ceil
```
- MXINT6
```
CUDA_VISIBLE_DEVICES=0 python3 cal_kl_ppl.py --model_path ./hf_models/Llama-3.2-1B --q1_x 6 --q1_w 6 --group_size 32 --e8_scale --e8_scale_op ceil
```
- MXFP6
```
CUDA_VISIBLE_DEVICES=0 python3 cal_kl_ppl.py --model_path ./hf_models/Llama-3.2-1B --quant_type fp --e_bit 2 --m_bit 3 --q1_x 6 --q1_w 6 --group_size 32 --e8_scale --e8_scale_op ceil
```
- MXINT4
```
CUDA_VISIBLE_DEVICES=0 python3 cal_kl_ppl.py --model_path ./hf_models/Llama-3.2-1B --q1_x 4 --q1_w 4 --group_size 32 --e8_scale --e8_scale_op ceil
```
- MXFP4
```
CUDA_VISIBLE_DEVICES=0 python3 cal_kl_ppl.py --model_path ./hf_models/Llama-3.2-1B --quant_type fp --e_bit 2 --m_bit 1 --q1_x 4 --q1_w 4 --group_size 32 --e8_scale --e8_scale_op ceil
```
- NVINT4
```
CUDA_VISIBLE_DEVICES=0 python3 cal_kl_ppl.py --model_path ./hf_models/Llama-3.2-1B --q1_x 4 --q1_w 4 --group_size 16 --scale_quant
```
- NVFP4
```
CUDA_VISIBLE_DEVICES=0 python3 cal_kl_ppl.py --model_path ./hf_models/Llama-3.2-1B --quant_type fp --e_bit 2 --m_bit 1 --q1_x 4 --q1_w 4 --group_size 16 --scale_quant
```



# Evaluate QSNR
We offer the scripts to evaluate the QSNR of quantized models across different low-bit formats. You can change `--model_path` to evaluate the KL divergence and perplexity of other models. You can also set `--rotate_dim 32` to introduce 32x32 random Hadamard rotation before quantization.
- MXINT8
```
CUDA_VISIBLE_DEVICES=0 python3 cal_qsnr.py \
--model_path ./hf_models/Llama-3.2-1B \
--bit 8 --e8_scale --e8_scale_op ceil \
--output_dir ./qsnr_results/llama3.2_1b/ \
--filename_prefix llama3.2_1b_base_mxint8
```

- MXFP8
```
CUDA_VISIBLE_DEVICES=0 python3 cal_qsnr.py \
--model_path ./hf_models/Llama-3.2-1B \
--bit 8 --e8_scale --e8_scale_op ceil \
--quant_type fp --e_bit 4 --m_bit 3 \
--output_dir ./qsnr_results/llama3.2_1b/ \
--filename_prefix llama3.2_1b_base_mxfp8
```

- MXINT6
```
CUDA_VISIBLE_DEVICES=0 python3 cal_qsnr.py \
--model_path ./hf_models/Llama-3.2-1B \
--bit 6 --e8_scale --e8_scale_op ceil \
--output_dir ./qsnr_results/llama3.2_1b/ \
--filename_prefix llama3.2_1b_base_mxint6
```

- MXFP6
```
CUDA_VISIBLE_DEVICES=0 python3 cal_qsnr.py \
--model_path ./hf_models/Llama-3.2-1B \
--bit 6 --e8_scale --e8_scale_op ceil \
--quant_type fp --e_bit 2 --m_bit 3 \
--output_dir ./qsnr_results/llama3.2_1b/ \
--filename_prefix llama3.2_1b_base_mxfp6
```

- MXINT4
```
CUDA_VISIBLE_DEVICES=1 python3 cal_qsnr.py \
--model_path ./hf_models/Llama-3.2-1B \
--bit 4 --e8_scale --e8_scale_op ceil \
--output_dir ./qsnr_results/llama3.2_1b/ \
--filename_prefix llama3.2_1b_base_mxint4
```

- MXFP4
```
CUDA_VISIBLE_DEVICES=2 python3 cal_qsnr.py \
--model_path ./hf_models/Llama-3.2-1B \
--bit 4 --e8_scale --e8_scale_op ceil \
--quant_type fp --e_bit 2 --m_bit 1 \
--output_dir ./qsnr_results/llama3.2_1b/ \
--filename_prefix llama3.2_1b_base_mxfp4
```

- NVINT4
```
CUDA_VISIBLE_DEVICES=0 python3 cal_qsnr.py \
--model_path ./hf_models/Llama-3.2-1B \
--bit 4 --scale_quant \
--output_dir ./qsnr_results/llama3.2_1b/ \
--filename_prefix llama3.2_1b_base_nvint4
```


- NVFP4
```
CUDA_VISIBLE_DEVICES=0 python3 cal_qsnr.py \
--model_path ./hf_models/Llama-3.2-1B \
--bit 4 --scale_quant \
--quant_type fp --e_bit 2 --m_bit 1 \
--output_dir ./qsnr_results/llama3.2_1b/ \
--filename_prefix llama3.2_1b_base_nvfp4
```


# Citation
```
TODO
```