import qai_hub as hub

compile_job = hub.submit_compile_job(
    model="potatoes.onnx",
    device=hub.Device("Snapdragon X Elite CRD"),
    input_specs={"input_1": (1, 256, 256, 3)},
    options="--target_runtime onnx"
)

print("Compile job submitted!")
print("View progress at:", compile_job.url)

# Wait for the job to actually finish
compile_job.wait()

if compile_job.success:
    compiled_model = compile_job.get_target_model()
    print("Compiled model ready.")
else:
    print("Compile job FAILED. Check the dashboard for details:", compile_job.url)