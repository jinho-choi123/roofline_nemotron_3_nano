#!/bin/bash

set -euo pipefail

# Install Nsight Compute CLI from NVIDIA .run installer.
INSTALL_URL="${INSTALL_URL:-https://developer.nvidia.com/downloads/assets/tools/secure/nsight-compute/2024_1_1/nsight-compute-linux-2024.1.1.4-33998838.run}"
INSTALLER_PATH="${INSTALLER_PATH:-/tmp/nsight_compute_installer.run}"

curl -L "$INSTALL_URL" -o "$INSTALLER_PATH"
chmod +x "$INSTALLER_PATH"
"$INSTALLER_PATH"

# After installation, the program is installed at /usr/local/NVIDIA-Nsight-Compute-2024.1

ln -sf /usr/local/NVIDIA-Nsight-Compute-2024.1/ncu /bin/ncu