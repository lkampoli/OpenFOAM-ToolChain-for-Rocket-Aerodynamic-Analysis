import subprocess

def run_executable_with_wine(executable_path):
    try:
        # Command to run the executable with Wine
        command = ['wine', executable_path]

        # Run the command
        subprocess.run(command, check=True)
        print(f'Successfully ran {executable_path} with Wine.')

    except subprocess.CalledProcessError as e:
        print(f'An error occurred while running the executable: {e}')
    except FileNotFoundError:
        print('Wine is not installed or executable file not found.')

# Replace 'path/to/MD0311.exe' with the actual path to your executable
run_executable_with_wine('./MD0311.exe')

# Run another Python script
subprocess.run(['python', 'plot_results.py'])
