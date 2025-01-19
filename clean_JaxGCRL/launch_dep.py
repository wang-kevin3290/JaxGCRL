import subprocess
from datetime import datetime

num_dep = 12  # the number of subjobs in the dependency chain to launch
epochs = [50] * 12  # [10] * 5 + [20] * 3
assert len(epochs) == num_dep

print(f"\nLaunching chain of {num_dep} jobs at {datetime.now().strftime('%H:%M:%S')}")
print(f"Epochs per job: {epochs}")

prev_sid = None
for i, num_epochs in enumerate(epochs):
    if i == 0:
        cmd = f"sbatch auto_checkpoint.slurm 0 0 {num_epochs} {num_epochs * 1_000_000}"
    else:
        cmd = f"sbatch --dependency=afterok:{prev_sid} auto_checkpoint.slurm 1 {prev_sid} {num_epochs} {num_epochs * 1_000_000}"
    
    print(f"\nJob {i+1}/{num_dep}")
    print(f"Command: {cmd}")
    
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    assert output.returncode == 0, f"sbatch failed with rc={output.returncode}, stderr={output.stderr}"
    prev_sid = output.stdout.split()[-1]
    assert output.stdout.strip() == f"Submitted batch job {prev_sid}"
    
    print(f"→ Submitted as job ID {prev_sid} ({num_epochs} epochs)")

print(f"\nAll jobs submitted successfully")