#!/usr/bin/env bash

# Robust, idempotent setup script for development environments.
# Usage: ./setup.sh [--distro ubuntu|arch] [--mode cpu|gpu] [--python 3.12.7] [--user USERNAME]

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults (override with flags)
PYTHON_VERSION="3.12.7"
DISTRO=""
MODE="cpu" # cpu or gpu
USER_NAME="$(id -un || true)"
NONINTERACTIVE=false

log() { printf "[INFO] %s\n" "$*"; }
warn() { printf "[WARN] %s\n" "$*"; }
fail() { printf "[ERROR] %s\n" "$*"; exit 1; }

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --distro=ubuntu|arch     Force distro (auto-detected by default)
  --mode=cpu|gpu           Install CPU or GPU dependencies (default: cpu)
  --python=VERSION         Python version for pyenv (default: ${PYTHON_VERSION})
  --user=USERNAME          Username to modify user-specific files (default: current user)
  --non-interactive        Run without prompts
  --help                   Show this help and exit
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help) usage; exit 0;;
      --distro=*) DISTRO="${1#*=}"; shift;;
      --distro) DISTRO="$2"; shift 2;;
      --mode=*) MODE="${1#*=}"; shift;;
      --mode) MODE="$2"; shift 2;;
      --python=*) PYTHON_VERSION="${1#*=}"; shift;;
      --python) PYTHON_VERSION="$2"; shift 2;;
      --user=*) USER_NAME="${1#*=}"; shift;;
      --user) USER_NAME="$2"; shift 2;;
      --non-interactive) NONINTERACTIVE=true; shift;;
      --*) warn "Unknown option: $1"; usage; exit 2;;
      *) break;;
    esac
  done
}

detect_distro() {
  if [[ -n "$DISTRO" ]]; then
    log "Using provided distro: $DISTRO"
    return
  fi
  if command -v lsb_release >/dev/null 2>&1; then
    ID=$(lsb_release -si | tr '[:upper:]' '[:lower:]') || true
  elif [[ -f /etc/os-release ]]; then
    ID=$(awk -F= '/^ID=/{gsub(/\"/,"",$2); print tolower($2)}' /etc/os-release)
  else
    ID="unknown"
  fi
  case "$ID" in
    ubuntu|debian) DISTRO=ubuntu ;;
    arch) DISTRO=arch ;;
    *) DISTRO="$ID" ;;
  esac
  log "Detected distro: $DISTRO"
}

ensure_sudo() {
  if [[ $EUID -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      SUDO='sudo'
    else
      warn "No sudo available; some operations may fail."
      SUDO=''
    fi
  else
    SUDO=''
  fi
}

install_system_packages() {
  log "Installing system packages for $DISTRO ($MODE)"
  if [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
    $SUDO apt update
    COMMON=(git curl build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
      libncurses5-dev libgdbm-dev libnss3-dev libffi-dev liblzma-dev wget xz-utils tk-dev)
    $SUDO apt install -y "${COMMON[@]}" python3-venv python3-distutils git-lfs || fail "apt install failed"
    if [[ "$MODE" == "gpu" ]]; then
      log "Please ensure NVIDIA drivers and CUDA are installed separately for GPU support."
    fi
  elif [[ "$DISTRO" == "arch" ]]; then
    $SUDO pacman -Syu --noconfirm
    $SUDO pacman -S --needed --noconfirm base-devel openssl zlib xz bzip2 libffi readline sqlite tk ncurses curl git-lfs
    if [[ "$MODE" == "gpu" ]]; then
      log "Please ensure NVIDIA drivers are installed via pacman/AUR for GPU support."
    fi
  else
    warn "Unsupported or unknown distro: $DISTRO. Skipping system package installation."
  fi
}

setup_pyenv() {
  if command -v pyenv >/dev/null 2>&1; then
    log "pyenv already installed: $(pyenv --version 2>/dev/null || true)"
  else
    log "Installing pyenv"
    curl https://pyenv.run | bash || fail "pyenv install script failed"
  fi

  PROFILE_FILE="${HOME}/.bashrc"
  if ! grep -q 'PYENV_ROOT' "$PROFILE_FILE" 2>/dev/null; then
    cat >> "$PROFILE_FILE" <<'EOF'
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi
EOF
    log "Appended pyenv initialization to $PROFILE_FILE"
  fi

  # shellcheck disable=SC1090
  source "$PROFILE_FILE" || true

  if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
    log "Installing Python $PYTHON_VERSION via pyenv (may take several minutes)"
    pyenv install -s "$PYTHON_VERSION" || fail "pyenv install failed"
  else
    log "Python $PYTHON_VERSION already installed in pyenv"
  fi

  # Use a project-specific pyenv version file
  (cd "$REPO_ROOT" && pyenv local "$PYTHON_VERSION") || warn "pyenv local failed"
}

create_and_activate_venv() {
  VENV_DIR="$REPO_ROOT/venv${PYTHON_VERSION}"
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtualenv at $VENV_DIR"
    pyenv which python 2>/dev/null || true
    python -m venv "$VENV_DIR" || fail "Failed to create venv"
  else
    log "Virtualenv already exists at $VENV_DIR"
  fi

  # Activate venv for the rest of this script
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
}

install_python_requirements() {
  cd "$REPO_ROOT"
  git lfs install --local || warn "git-lfs not available"
  git lfs pull || warn "git-lfs pull failed or no LFS objects"

  REQ_CPU="setup/requirements-cpu.txt"
  REQ_GPU="setup/requirements-gpu.txt"

  if [[ "$MODE" == "gpu" && -f "$REQ_GPU" ]]; then
    log "Installing GPU requirements from $REQ_GPU"
    pip install -r "$REQ_GPU"
  elif [[ -f "$REQ_CPU" ]]; then
    log "Installing CPU requirements from $REQ_CPU"
    pip install -r "$REQ_CPU"
  else
    warn "No matching requirements file found for mode=$MODE. Skipping pip installs."
  fi
}

main() {
  parse_args "$@"
  detect_distro
  ensure_sudo
  install_system_packages
  setup_pyenv
  create_and_activate_venv
  install_python_requirements

  log "Setup complete. To activate the environment later run:"
  log "  source $REPO_ROOT/venv${PYTHON_VERSION}/bin/activate"
}

main "$@"