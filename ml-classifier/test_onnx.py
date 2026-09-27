import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("potatoes.onnx")
dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)

result = session.run(None, {"input_1": dummy_input})
print("Output shape:", result[0].shape)
print("Output values:", result[0])