import sys
import subprocess
import os
import re
import time
def modify_and_submit_slurm(file_name, depth, skip):
    # Read the original slurm file
    slurm_file = f"{file_name}.slurm"
    with open(slurm_file, 'r') as f:
        lines = f.readlines()
    
    # Create modified content
    modified_lines = []
    for line in lines:
        # Replace job name by removing any trailing numbers and adding skip
        if line.startswith('#SBATCH --job-name='):
            base_name = re.sub(r'\d+$', '', line.strip())  # Remove trailing numbers
            line = f'{base_name}{skip}\n'
        # Replace DEPTH
        elif 'DEPTH=' in line:
            line = f'DEPTH={depth}           # 4, 8, 16 , 32\n'
        # Replace SKIP
        elif 'SKIP=' in line:
            line = f'SKIP={skip}            # 2, 3, 4, 8\n'
        modified_lines.append(line)
    
    # Write modified content back to file
    with open(slurm_file, 'w') as f:
        f.writelines(modified_lines)
    
    # Submit the job
    subprocess.run(['sbatch', slurm_file])
    print(f"Submitted job with DEPTH={depth}, SKIP={skip}")
    time.sleep(4)

def main():
    if len(sys.argv) != 4:
        print("Usage: python runTests.py <fileName> <depths> <skips>")
        print("Example: python runTests.py humanoid '4,8,16,32' '2,3,4,8'")
        sys.exit(1)

    file_name = sys.argv[1]
    depths = [int(d) for d in sys.argv[2].split(',')]
    skips = [int(s) for s in sys.argv[3].split(',')]

    # Check if slurm file exists
    if not os.path.exists(f"{file_name}.slurm"):
        print(f"Error: {file_name}.slurm not found")
        sys.exit(1)

    # Double loop over depths and skips
    for skip in skips:
        for depth in depths:
            modify_and_submit_slurm(file_name, depth, skip)

if __name__ == "__main__":
    main()