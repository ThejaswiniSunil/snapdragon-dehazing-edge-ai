import qai_hub as hub

compile_job = hub.get_job("jpxlzzl1p")  # your compile job ID from the dashboard
target_model = compile_job.get_target_model()

profile_job = hub.submit_profile_job(
    model=target_model,
    device=hub.Device("Snapdragon X Elite CRD"),
)

print("NPU profile job submitted!")
print("View progress at:", profile_job.url)

profile_job.wait()

if profile_job.success:
    print("NPU profiling complete — check the dashboard.")
else:
    print("Profile job FAILED. Check:", profile_job.url)