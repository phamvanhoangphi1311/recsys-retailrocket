#!/bin/bash
# ============================================
# SETUP DEV ENVIRONMENT — Ubuntu
# Chạy: chmod +x setup_ubuntu.sh && ./setup_ubuntu.sh
# ============================================

echo "=== 1. Update system ==="
sudo apt update && sudo apt upgrade -y

echo "=== 2. Install Git ==="
sudo apt install -y git
git --version

echo "=== 3. Install Docker ==="
# Remove old versions
sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null

# Install Docker
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Run Docker without sudo
sudo usermod -aG docker $USER
echo ">>> Log out and log back in for Docker group to take effect"

echo "=== 4. Verify ==="
docker --version
git --version

echo ""
echo "=== DONE ==="
echo "Next: log out, log back in, then run 'docker run hello-world' to verify"