#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Run this script with sudo from your terminal."
    exit 1
fi

source /etc/os-release
if [[ $ID != fedora ]]; then
    echo "This installer is intended for Fedora."
    exit 1
fi

running_kernel=$(uname -r)

# The NVIDIA RPM Fusion repository is already enabled on this machine.
# Exclude the unrelated ChatGPT repository, whose metadata signing key is
# currently missing. Package signature verification remains enabled.
dnf --disablerepo=openai-chatgpt install -y \
    "kernel-devel-${running_kernel}" \
    akmods akmod-nvidia-open xorg-x11-drv-nvidia-cuda mokutil

if [[ ! -f /etc/pki/akmods/certs/public_key.der ]]; then
    kmodgenca -a
fi

akmods --force --rebuild --kernels "$running_kernel"
depmod -a "$running_kernel"
module_signer=$(modinfo -k "$running_kernel" -F signer nvidia)
if [[ -z $module_signer ]]; then
    echo "NVIDIA kernel module was not signed; stop before rebooting."
    exit 1
fi
echo "NVIDIA module signer: $module_signer"
dracut --force --kver "$running_kernel"

if [[ $(mokutil --sb-state) == *"SecureBoot enabled"* ]]; then
    # Some mokutil versions return nonzero even while reporting an enrolled key.
    # Check its explicit result instead of interpreting every nonzero exit as
    # a request for enrollment; errors must not become enrollment requests.
    enrollment_result=$(LC_ALL=C mokutil --test-key /etc/pki/akmods/certs/public_key.der 2>&1) || true
    printf '%s\n' "$enrollment_result"
    case "$enrollment_result" in
        *"is already enrolled"*)
            echo "Signing key already enrolled; no enrollment step is needed."
            ;;
        *"is not enrolled"*)
            echo "Choose a temporary enrollment password at the next prompt."
            mokutil --import /etc/pki/akmods/certs/public_key.der
            echo "After reboot: Enroll MOK -> Continue -> Yes, then enter that password."
            ;;
        *)
            echo "Could not determine signing-key enrollment. Stop and inspect the result above."
            exit 1
            ;;
    esac
fi

echo "Driver installation and kernel-module build completed."
echo "Reboot when ready, then run nvidia-smi and the project's GPU check."
