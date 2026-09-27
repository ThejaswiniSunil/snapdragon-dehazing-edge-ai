import torch
import onnxruntime as ort
import numpy as np
from dehaze_net import ResnetGenerator
import torch.nn as nn

dummy_input = torch.randn(1, 3, 256, 256)

# Run through the original PyTorch model
model = ResnetGenerator(input_nc=3, output_nc=3, norm_layer=nn.InstanceNorm2d)
model.load_state_dict(torch.load("remove_hazy_model_256x256.pth", map_location="cpu"))
model.eval()
with torch.no_grad():
    torch_output = model(dummy_input).numpy()

# Run through the ONNX model
session = ort.InferenceSession("dehaze.onnx")
onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

# Compare
diff = np.abs(torch_output - onnx_output).max()
print("Output shape:", onnx_output.shape)
print("Max difference between PyTorch and ONNX outputs:", diff)