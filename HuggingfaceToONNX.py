from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from onnxruntime.transformers import optimizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Convert to ONNX
dummy_input = tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"]

torch.onnx.export(
    model,
    (dummy_input,),
    f"{model_name}.onnx",
    export_params=True,
    opset_version=12,
    input_names=["input_ids"],
    output_names=["output"],
)
