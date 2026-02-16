import subprocess


# def run_pip_command(command):
#     try:
#         subprocess.check_call(command, shell=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Error executing command: {command}")
#         print(f"Error message: {str(e)}")
#
#
# # List of pip commands
# pip_commands = [
#     # "pip install setuptools==58.0.0",
#     # "pip install wheel==0.37.1",
#     # "pip install numpy==1.23.0",
#     # "pip install typing-extensions==3.7.4.3",
#     # "pip install protobuf==3.19.4",
#     # "pip install tensorflow==2.8.3",
#     # "pip install tensorflow-gpu==2.8.3",
#     # "pip install h5py==3.1.0",
#     # "pip install absl-py==1.0.0",
#     # "pip install tflite-model-maker==0.3.4",
#     # "pip install tensorflow-datasets==4.4.0",
#     # "pip install tensorflow-addons==0.16.1",
#     # "pip install tensorflowjs==3.18.0",
#     # "pip install tf-models-official==2.3.0",
#     # "pip install pandas==1.4.4",
#     # "pip install matplotlib==3.4.3",
#     # "pip install librosa==0.8.1",
#     # "pip install scipy==1.5.4",
#     # "pip install Pillow==9.0.0",
#     # "pip install opencv-python-headless==4.5.5.64",
#     # "pip install typeguard==2.13.3",
#     # "pip install flatbuffers==1.12",
#     # "pip install Cython==0.29.24",
#     # "pip install numba==0.53.0",
#     # "pip check",
#     # "pip install typing-extensions==3.10.0.0",
#     # "pip install numpy==1.23.0",
#     # "pip install pycocotools",
#     "pip list"
# ]
#
# # Execute pip commands
# for command in pip_commands:
#     print(f"Executing: {command}")
#     run_pip_command(command)
#
# print("All commands executed.")



import subprocess
import sys


def run(cmd):
    print(f">> {cmd}")
    result = subprocess.run(
        [sys.executable, "-m"] + cmd.split(),
        capture_output=True,
        text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode


def main():
    # Remove all three opencv packages - they conflict with each other
    run("pip uninstall opencv-python -y")
    run("pip uninstall opencv-python-headless -y")
    run("pip uninstall opencv-contrib-python -y")

    # numpy 2.0.2 is what broke everything.
    # TF 2.8.3 needs numpy <1.24 at runtime for the bfloat16 binding.
    # numba 0.53.0 needs numpy >=1.17,<1.24.
    # 1.23.5 is the highest 1.23.x patch, satisfies both.
    run("pip install numpy==1.23.5")

    # opencv-python-headless 4.5.5.64 is what your original requirements had
    # and it was working before. The "contrib" variant you tried to install
    # does NOT ship with CUDA support via pip - it needs to be built from
    # source for that. There is no pip-installable opencv with CUDA for
    # Python 3.9 that works with numpy 1.23.x.
    run("pip install opencv-python-headless==4.5.5.64")

    # Verify
    run("pip check")


if __name__ == "__main__":
    main()