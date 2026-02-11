import torch
from model import TCN

model = TCN()
model.load_state_dict(torch.load("experiments/exp_001/model.pth"))
model.eval()

dummy = torch.randn(1,48,1)
torch.onnx.export(
    model,
    dummy,
    "experiments/exp_001/model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}},
    opset_version=14,
)
print("Exported ONNX model")