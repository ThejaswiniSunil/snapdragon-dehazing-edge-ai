import torch
from dehaze_net import ResnetGenerator
import torch.nn as nn

# Rebuild the exact architecture used when the weights were saved
model = ResnetGenerator(input_nc=3, output_nc=3, norm_layer=nn.InstanceNorm2d)
model.load_state_dict(torch.load("remove_hazy_model_256x256.pth", map_location="cpu"))
model.eval()

# Dummy input matching the model's expected shape: batch, channels, height, width
dummy_input = torch.randn(1, 3, 256, 256)

torch.onnx.export(
    model,
    dummy_input,
    "dehaze.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamo=False
)

print("Exported dehaze.onnx successfully.")