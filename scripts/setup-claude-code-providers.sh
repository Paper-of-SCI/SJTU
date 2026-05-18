#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${CLAUDE_CODE_PROVIDER_CONFIG_DIR:-$HOME/.config/claude-code}"
BIN_DIR="${CLAUDE_CODE_PROVIDER_BIN_DIR:-$HOME/.local/bin}"
DEFAULT_PROVIDER="deepseek"
QWEN_ENDPOINT="cn"
WRITE_KEYS=0
CONFIGURE_VSCODE=1
SHELL_RC=""

usage() {
    cat <<'USAGE'
Configure Claude Code to use DeepSeek and Alibaba Qwen through Anthropic-compatible APIs.

Usage:
  setup-claude-code-providers.sh [options]

Options:
  --default-provider <name>   deepseek, deepseek-flash, qwen, or qwen-flash
                              default: deepseek
  --qwen-endpoint <value>     cn, intl, coding, or a full https://... URL
                              default: cn
  --write-keys                Prompt for DeepSeek/Qwen API keys and store them locally
  --skip-vscode               Do not modify VS Code Claude extension settings
  --rc <path>                 Shell rc file to patch, default: ~/.bashrc or ~/.zshrc
  --config-dir <path>         Config output dir, default: ~/.config/claude-code
  --bin-dir <path>            Wrapper output dir, default: ~/.local/bin
  -h, --help                  Show this help

After setup, start a new shell or run:
  source ~/.bashrc

Commands added:
  ccds   DeepSeek Pro
  ccdf   DeepSeek Flash
  ccqw   Qwen Plus
  ccqf   Qwen Flash
  cc @qwen / cc @deepseek / cc @qwen-flash / cc @deepseek-flash
  cc-provider qwen|deepseek|qwen-flash|deepseek-flash

API keys are stored in:
  ~/.config/claude-code/deepseek.token
  ~/.config/claude-code/qwen.token
USAGE
}

die() {
    echo "error: $*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --default-provider)
            [ "$#" -ge 2 ] || die "--default-provider needs a value"
            DEFAULT_PROVIDER="$2"
            shift 2
            ;;
        --qwen-endpoint)
            [ "$#" -ge 2 ] || die "--qwen-endpoint needs a value"
            QWEN_ENDPOINT="$2"
            shift 2
            ;;
        --write-keys)
            WRITE_KEYS=1
            shift
            ;;
        --skip-vscode)
            CONFIGURE_VSCODE=0
            shift
            ;;
        --rc)
            [ "$#" -ge 2 ] || die "--rc needs a path"
            SHELL_RC="$2"
            shift 2
            ;;
        --config-dir)
            [ "$#" -ge 2 ] || die "--config-dir needs a path"
            CONFIG_DIR="$2"
            shift 2
            ;;
        --bin-dir)
            [ "$#" -ge 2 ] || die "--bin-dir needs a path"
            BIN_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

case "$DEFAULT_PROVIDER" in
    deepseek|deepseek-flash|qwen|qwen-flash) ;;
    *) die "invalid default provider: $DEFAULT_PROVIDER" ;;
esac

qwen_base_url() {
    case "$QWEN_ENDPOINT" in
        cn) echo "https://dashscope.aliyuncs.com/apps/anthropic" ;;
        intl|sg|singapore) echo "https://dashscope-intl.aliyuncs.com/apps/anthropic" ;;
        coding|coding-plan) echo "https://coding.dashscope.aliyuncs.com/apps/anthropic" ;;
        https://*) echo "$QWEN_ENDPOINT" ;;
        *) die "invalid qwen endpoint: $QWEN_ENDPOINT" ;;
    esac
}

detect_shell_rc() {
    if [ -n "$SHELL_RC" ]; then
        echo "$SHELL_RC"
        return
    fi
    case "${SHELL:-}" in
        */zsh) echo "$HOME/.zshrc" ;;
        *) echo "$HOME/.bashrc" ;;
    esac
}

write_env_files() {
    local qwen_url
    qwen_url="$(qwen_base_url)"

    mkdir -p "$CONFIG_DIR"
    chmod 700 "$CONFIG_DIR"

    cat > "$CONFIG_DIR/deepseek.env" <<'EOF'
# Claude Code -> DeepSeek Anthropic-compatible API.
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_MODEL="deepseek-v4-pro[1m]"
export ANTHROPIC_SMALL_FAST_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_EFFORT_LEVEL="max"

unset ANTHROPIC_API_KEY
unset ANTHROPIC_DEFAULT_OPUS_MODEL
unset ANTHROPIC_DEFAULT_SONNET_MODEL
unset ANTHROPIC_DEFAULT_HAIKU_MODEL

token_dir="${CLAUDE_CODE_PROVIDER_CONFIG_DIR:-$HOME/.config/claude-code}"
if [ -r "$token_dir/deepseek.token" ]; then
    export ANTHROPIC_AUTH_TOKEN="$(tr -d '\r\n' < "$token_dir/deepseek.token")"
fi
EOF

    cat > "$CONFIG_DIR/deepseek-flash.env" <<'EOF'
# Claude Code -> DeepSeek Flash.
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_MODEL="deepseek-v4-flash"
export ANTHROPIC_SMALL_FAST_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_EFFORT_LEVEL="max"

unset ANTHROPIC_API_KEY
unset ANTHROPIC_DEFAULT_OPUS_MODEL
unset ANTHROPIC_DEFAULT_SONNET_MODEL
unset ANTHROPIC_DEFAULT_HAIKU_MODEL

token_dir="${CLAUDE_CODE_PROVIDER_CONFIG_DIR:-$HOME/.config/claude-code}"
if [ -r "$token_dir/deepseek.token" ]; then
    export ANTHROPIC_AUTH_TOKEN="$(tr -d '\r\n' < "$token_dir/deepseek.token")"
fi
EOF

    cat > "$CONFIG_DIR/qwen.env" <<EOF
# Claude Code -> Alibaba Cloud Model Studio/Qwen Anthropic-compatible API.
export ANTHROPIC_BASE_URL="$qwen_url"
export ANTHROPIC_MODEL="qwen3.6-plus"
export ANTHROPIC_SMALL_FAST_MODEL="qwen3.6-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="qwen3.6-plus"
export CLAUDE_CODE_EFFORT_LEVEL="max"

unset ANTHROPIC_API_KEY
unset ANTHROPIC_DEFAULT_OPUS_MODEL
unset ANTHROPIC_DEFAULT_SONNET_MODEL
unset ANTHROPIC_DEFAULT_HAIKU_MODEL

token_dir="\${CLAUDE_CODE_PROVIDER_CONFIG_DIR:-\$HOME/.config/claude-code}"
if [ -r "\$token_dir/qwen.token" ]; then
    export ANTHROPIC_AUTH_TOKEN="\$(tr -d '\\r\\n' < "\$token_dir/qwen.token")"
fi
EOF

    cat > "$CONFIG_DIR/qwen-flash.env" <<EOF
# Claude Code -> Alibaba Cloud Model Studio/Qwen Flash.
export ANTHROPIC_BASE_URL="$qwen_url"
export ANTHROPIC_MODEL="qwen3.6-flash"
export ANTHROPIC_SMALL_FAST_MODEL="qwen3.6-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="qwen3.6-flash"
export CLAUDE_CODE_EFFORT_LEVEL="max"

unset ANTHROPIC_API_KEY
unset ANTHROPIC_DEFAULT_OPUS_MODEL
unset ANTHROPIC_DEFAULT_SONNET_MODEL
unset ANTHROPIC_DEFAULT_HAIKU_MODEL

token_dir="\${CLAUDE_CODE_PROVIDER_CONFIG_DIR:-\$HOME/.config/claude-code}"
if [ -r "\$token_dir/qwen.token" ]; then
    export ANTHROPIC_AUTH_TOKEN="\$(tr -d '\\r\\n' < "\$token_dir/qwen.token")"
fi
EOF

    printf "%s\n" "$DEFAULT_PROVIDER" > "$CONFIG_DIR/default-provider"
    chmod 600 "$CONFIG_DIR"/*.env "$CONFIG_DIR/default-provider"
}

write_bin_helpers() {
    mkdir -p "$BIN_DIR"

    cat > "$BIN_DIR/setup-claude-provider-key" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

provider="${1:-}"
case "$provider" in
    deepseek|qwen) ;;
    "")
        echo "usage: setup-claude-provider-key deepseek|qwen"
        exit 2
        ;;
    *)
        echo "unknown provider: $provider" >&2
        echo "available providers: deepseek, qwen" >&2
        exit 2
        ;;
esac

config_dir="${CLAUDE_CODE_PROVIDER_CONFIG_DIR:-$HOME/.config/claude-code}"
token_file="$config_dir/$provider.token"

mkdir -p "$config_dir"
chmod 700 "$config_dir"

printf "%s API Key: " "$provider"
IFS= read -r -s token
printf "\n"

if [ -z "$token" ]; then
    echo "not written: empty API key"
    exit 1
fi

umask 077
printf "%s\n" "$token" > "$token_file"
chmod 600 "$token_file"
echo "wrote $token_file"
EOF

    cat > "$BIN_DIR/setup-claude-deepseek-key" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/setup-claude-provider-key" deepseek
EOF

    cat > "$BIN_DIR/claude-code-provider-wrapper" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

config_dir="${CLAUDE_CODE_PROVIDER_CONFIG_DIR:-$HOME/.config/claude-code}"
provider="${CLAUDE_PROVIDER:-}"
if [ -z "$provider" ] && [ -r "$config_dir/default-provider" ]; then
    provider="$(tr -d '\r\n' < "$config_dir/default-provider")"
fi
provider="${provider:-deepseek}"

env_file="$config_dir/${provider}.env"
if [ ! -r "$env_file" ]; then
    echo "missing Claude Code provider config: $env_file" >&2
    exit 2
fi

. "$env_file"

real_claude="${CLAUDE_REAL_BIN:-}"
if [ -z "$real_claude" ]; then
    real_claude="$(command -v claude || true)"
fi
if [ -z "$real_claude" ] || [ ! -x "$real_claude" ]; then
    echo "cannot find executable claude; install Claude Code first or set CLAUDE_REAL_BIN" >&2
    exit 127
fi

exec "$real_claude" "$@"
EOF

    chmod 700 "$BIN_DIR/setup-claude-provider-key" \
        "$BIN_DIR/setup-claude-deepseek-key" \
        "$BIN_DIR/claude-code-provider-wrapper"
}

write_statusline() {
    local claude_dir="$HOME/.claude"
    local script="$claude_dir/statusline-usage.py"
    mkdir -p "$claude_dir"

    cat > "$script" <<'EOF'
#!/usr/bin/env python3
import json
import os
import sys


def compact_tokens(value):
    try:
        n = int(value or 0)
    except (TypeError, ValueError):
        n = 0
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def provider_name():
    provider = os.environ.get("CLAUDE_PROVIDER", "")
    if not provider:
        default_path = os.path.expanduser("~/.config/claude-code/default-provider")
        try:
            with open(default_path, "r", encoding="utf-8") as handle:
                provider = handle.read().strip()
        except OSError:
            provider = ""
    return provider or "provider?"


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    model = (data.get("model") or {}).get("display_name") or (data.get("model") or {}).get("id") or "model?"
    context = data.get("context_window") or {}
    usage = context.get("current_usage") or {}
    cost = data.get("cost") or {}

    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cache_create = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    fresh = input_tokens + cache_create
    total_input = fresh + cache_read
    hit_rate = int(round(cache_read * 100 / total_input)) if total_input > 0 else 0

    try:
        context_text = f"{float(context.get('used_percentage')):.1f}%"
    except (TypeError, ValueError):
        context_text = "--%"

    try:
        cost_text = f"${float(cost.get('total_cost_usd')):.4f}"
    except (TypeError, ValueError):
        cost_text = "$--"

    print(
        " | ".join(
            [
                f"{provider_name()}:{model}",
                f"ctx {context_text}",
                f"in {compact_tokens(input_tokens)}",
                f"out {compact_tokens(output_tokens)}",
                f"cache+ {compact_tokens(cache_create)}",
                f"cache hit {compact_tokens(cache_read)} ({hit_rate}%)",
                f"cost {cost_text}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EOF

    chmod 700 "$script"

    local settings="$claude_dir/settings.json"
    python3 - "$settings" "$script" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
script = sys.argv[2]
if path.exists():
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {}
else:
    data = {}
data["statusLine"] = {
    "type": "command",
    "command": script,
    "padding": 1,
}
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

patch_shell_rc() {
    local rc marker_start marker_end temp
    rc="$(detect_shell_rc)"
    marker_start="# >>> claude code providers >>>"
    marker_end="# <<< claude code providers <<<"
    mkdir -p "$(dirname "$rc")"
    touch "$rc"

    temp="$(mktemp)"
    awk -v start="$marker_start" -v end="$marker_end" '
        $0 == start {skip=1; next}
        $0 == end {skip=0; next}
        !skip {print}
    ' "$rc" > "$temp"

    cat >> "$temp" <<'EOF'

# >>> claude code providers >>>
# Run Claude Code through a selected Anthropic-compatible provider.
# Store keys with:
#   setup-claude-provider-key deepseek
#   setup-claude-provider-key qwen
__claude_provider_run() {
    local provider="$1"
    shift
    local config_dir="${CLAUDE_CODE_PROVIDER_CONFIG_DIR:-$HOME/.config/claude-code}"
    local env_file="$config_dir/${provider}.env"

    if [ ! -r "$env_file" ]; then
        echo "missing Claude Code provider config: $env_file" >&2
        return 2
    fi

    ( . "$env_file"; command claude "$@" )
}

claude() {
    local provider="${CLAUDE_PROVIDER:-deepseek}"
    case "${1:-}" in
        @deepseek|--deepseek)
            provider="deepseek"
            shift
            ;;
        @deepseek-flash|@ds-flash|--deepseek-flash|--ds-flash)
            provider="deepseek-flash"
            shift
            ;;
        @qwen|@ali|--qwen|--ali)
            provider="qwen"
            shift
            ;;
        @qwen-flash|@qw-flash|--qwen-flash|--qw-flash)
            provider="qwen-flash"
            shift
            ;;
        --provider)
            provider="${2:-}"
            shift 2
            ;;
    esac

    __claude_provider_run "$provider" "$@"
}

cc() {
    claude "$@"
}

ccds() {
    __claude_provider_run deepseek "$@"
}

ccdf() {
    __claude_provider_run deepseek-flash "$@"
}

ccqw() {
    __claude_provider_run qwen "$@"
}

ccqf() {
    __claude_provider_run qwen-flash "$@"
}

cc-provider() {
    case "${1:-}" in
        deepseek|deepseek-flash|qwen|qwen-flash)
            export CLAUDE_PROVIDER="$1"
            local config_dir="${CLAUDE_CODE_PROVIDER_CONFIG_DIR:-$HOME/.config/claude-code}"
            mkdir -p "$config_dir"
            printf "%s\n" "$1" > "$config_dir/default-provider"
            echo "Claude Code default provider for this terminal: $CLAUDE_PROVIDER"
            ;;
        "")
            echo "${CLAUDE_PROVIDER:-deepseek}"
            ;;
        *)
            echo "available providers: deepseek, deepseek-flash, qwen, qwen-flash" >&2
            return 2
            ;;
    esac
}
# <<< claude code providers <<<
EOF

    mv "$temp" "$rc"
    echo "patched shell rc: $rc"
}

configure_vscode_settings() {
    [ "$CONFIGURE_VSCODE" -eq 1 ] || return 0

    local wrapper="$BIN_DIR/claude-code-provider-wrapper"
    local candidates=(
        "$HOME/.vscode-server/data/Machine/settings.json"
        "$HOME/.config/Code/User/settings.json"
        "$HOME/.vscode/settings.json"
    )
    local found=0
    local path

    for path in "${candidates[@]}"; do
        if [ -f "$path" ]; then
            found=1
            python3 - "$path" "$wrapper" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
wrapper = sys.argv[2]
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    raise SystemExit(f"skip invalid JSON settings: {path}")
data["claudeCode.claudeProcessWrapper"] = wrapper
path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
print(f"patched VS Code settings: {path}")
PY
        fi
    done

    if [ "$found" -eq 0 ]; then
        echo "no VS Code settings file found; skipped VS Code plugin wrapper"
    fi
}

prompt_key() {
    local provider="$1"
    "$BIN_DIR/setup-claude-provider-key" "$provider"
}

main() {
    write_env_files
    write_bin_helpers
    write_statusline
    patch_shell_rc
    configure_vscode_settings

    if [ "$WRITE_KEYS" -eq 1 ]; then
        prompt_key deepseek
        prompt_key qwen
    fi

    echo
    echo "done."
    echo "Next:"
    echo "  source $(detect_shell_rc)"
    echo "  setup-claude-provider-key deepseek"
    echo "  setup-claude-provider-key qwen"
    echo "  ccds -p 'reply OK only'"
    echo "  ccqw -p 'reply OK only'"
}

main "$@"
