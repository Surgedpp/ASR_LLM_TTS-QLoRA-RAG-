modelpath = "/mnt/d/CodeX_WS/LLM_LoRA/cockpit_lf/export/deepseek_r1_qwen_1_5b_cockpit_merged_from_qlora"
1.主要量化的参数设置：
ret = llm.load_huggingface(
    model=modelpath,
    model_lora=None,
    device="cuda",
    dtype="float16",
    custom_config=None,
    load_weight=True
)

dataset = "./data_quant.json"
qparams = None
target_platform = "rk3588"
optimization_level = 1
quantized_dtype = "w4a16_g64"
quantized_algorithm = "grq"
num_npu_core = 2

2.校准集的更改，更适合智能座舱场景
