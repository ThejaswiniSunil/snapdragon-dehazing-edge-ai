"""
Profiles the exported Llama-3.2-1B-Instruct QNN model on Qualcomm AI Hub's
hosted Snapdragon X Elite device, once on NPU and once on CPU, so the two
numbers can be compared directly in the README benchmark table.

Run this after run_export.sh has finished and ./export-output exists.

Note: adjust MODEL_PATH below if qai_hub_models names the compiled
artifact differently than assumed here — check ./export-output after
export completes.
"""

import qai_hub as hub

DEVICE = "Snapdragon X Elite CRD"
MODEL_PATH = "./export-output/model.bin"  # confirm actual filename after export

def run_profile(compute_unit: str):
    print(f"Submitting profile job on {compute_unit}...")
    job = hub.submit_profile_job(
        model=MODEL_PATH,
        device=hub.Device(DEVICE),
        options=f"--compute_unit {compute_unit}",
    )
    job.wait()
    results = job.download_profile()
    print(f"--- {compute_unit} results ---")
    print(results)
    return results

if __name__ == "__main__":
    npu_results = run_profile("npu")
    cpu_results = run_profile("cpu")

    print("\nDone. Paste the relevant latency/memory/compute-unit numbers")
    print("from above into the README.md benchmark table.")
