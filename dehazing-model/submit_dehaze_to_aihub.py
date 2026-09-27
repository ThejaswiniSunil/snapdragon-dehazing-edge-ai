import qai_hub as hub

compile_job = hub.submit_compile_job(
    model="dehaze.onnx",
    device=hub.Device("Snapdragon X Elite CRD"),
    input_specs={"input": (1, 3, 256, 256)},
    options="--target_runtime onnx"
)

print("Compile job submitted!")
print("View progress at:", compile_job.url)

compile_job.wait()

if compile_job.success:
    target_model = compile_job.get_target_model()
    print("Compiled model ready.")
else:
    print("Compile job FAILED. Check the dashboard:", compile_job.url)