import qai_hub as hub

compile_job = hub.get_job("jpxlzd69p")  # same compile job ID as before
target_model = compile_job.get_target_model()

profile_job_cpu = hub.submit_profile_job(
    model=target_model,
    device=hub.Device("Snapdragon X Elite CRD"),
    options="--compute_unit cpu"
)

print("CPU profile job submitted!")
print("View progress at:", profile_job_cpu.url)

profile_job_cpu.wait()

if profile_job_cpu.success:
    print("Profiling complete — check the dashboard for CPU-only latency.")
else:
    print("Profile job FAILED. Check:", profile_job_cpu.url)