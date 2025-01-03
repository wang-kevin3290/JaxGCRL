import sys
import subprocess
import os
import re
import time

# Define the mapping from abbreviations to full names
ARCH_TYPE_MAP = {
    'r': 'resnet',
    'nr': 'noresnet',
    'ro': 'resnetOrig',
    'im': 'identityMapping'
}

def modify_and_submit_slurm(file_name, depth, arch_abbrev):
    arch_type = ARCH_TYPE_MAP[arch_abbrev]
    
    # Read the original slurm file
    slurm_file = f"{file_name}.slurm"
    with open(slurm_file, 'r') as f:
        lines = f.readlines()
    
    # Create modified content
    modified_lines = []
    for line in lines:
        # Handle job name with special cases
        if line.startswith('#SBATCH --job-name='):
            current_name = line.split('=')[1].strip()
            if '_' in current_name:
                base_name = current_name.split('_')[0]
            elif current_name[-1].isdigit():
                # Strip the trailing number and add new suffix
                base_name = ''.join(c for c in current_name if not c.isdigit())
            else:
                base_name = current_name
            line = f'#SBATCH --job-name={base_name}_{arch_abbrev}{depth}\n'
        # Rest of the conditions remain the same
        elif 'DEPTH=' in line:
            line = f'DEPTH={depth}           # 4, 8, 16 , 32\n'
        elif 'SKIP=' in line:
            line = f'SKIP=4            # hardcoded to 4\n'
        elif '--resnet=' in line:
            resnet_pos = line.find('--resnet=')
            before_resnet = line[:resnet_pos + 9]
            rest_of_line = line[resnet_pos + 9:]
            next_space = rest_of_line.find(' ')
            if next_space == -1:
                line = f'{before_resnet}"{arch_type}"\n'
            else:
                line = f'{before_resnet}"{arch_type}"{rest_of_line[next_space:]}'
        elif 'WANDB_GROUP=' in line:
            line = f'WANDB_GROUP="{arch_abbrev}_{depth}"\n'
        modified_lines.append(line)
    
    # Write modified content back to file
    with open(slurm_file, 'w') as f:
        f.writelines(modified_lines)
    
    # Submit the job
    subprocess.run(['sbatch', slurm_file])

    print(f"Submitted job with DEPTH={depth}, ARCH_TYPE={arch_abbrev}")
    time.sleep(4)

def main():
    if len(sys.argv) != 4:
        print("Usage: python runTests.py <fileName> <depths> <arch_types>")
        print("Example: python runTests.py humanoid '4,8,16,32' 'r,nr,ro,im'")
        sys.exit(1)

    file_name = sys.argv[1]
    depths = [int(d) for d in sys.argv[2].split(',')]
    arch_abbrevs = sys.argv[3].split(',')

    # Validate architecture abbreviations
    for abbrev in arch_abbrevs:
        if abbrev not in ARCH_TYPE_MAP:
            print(f"Error: Invalid architecture type '{abbrev}'. Valid options are: {', '.join(ARCH_TYPE_MAP.keys())}")
            sys.exit(1)

    # Check if slurm file exists
    if not os.path.exists(f"{file_name}.slurm"):
        print(f"Error: {file_name}.slurm not found")
        sys.exit(1)

    # Double loop over depths and architecture types
    for arch_abbrev in arch_abbrevs:
        for depth in depths:
            modify_and_submit_slurm(file_name, depth, arch_abbrev)

if __name__ == "__main__":
    main()