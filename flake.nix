{
  description = "comp2211 labs project using uv2nix";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python313;
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
            ]
          )
      );

    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonSet = pythonSets.${system}.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv "hello-world-dev-env" workspace.deps.all;
        in
        {
          default = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
              pkgs.watchexec
              pkgs.gnugrep
              pkgs.util-linux
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              PY_WATCH_TTY="$(ps -o tty= -p $$ 2>/dev/null | tr -d '[:space:]')"
              PY_WATCH_TTY_DEVICE=""
              PY_WATCH_PID_FILE=""
              PY_WATCH_SHELL_PID=""
              PY_WATCH_TERMINAL_FREE=0
              if [ -n "$PY_WATCH_TTY" ] && [ "$PY_WATCH_TTY" != "?" ]; then
                PY_WATCH_TTY_DEVICE="/dev/$PY_WATCH_TTY"
                PY_WATCH_TTY_SLUG="''${PY_WATCH_TTY//\//_}"
                PY_WATCH_PID_FILE="$REPO_ROOT/.direnv/python-watcher.$PY_WATCH_TTY_SLUG.pid"
                PY_WATCH_SHELL_PID="$(
                  ps -t "$PY_WATCH_TTY" -o pid=,comm= 2>/dev/null |
                    awk '$2 == "fish" || $2 == "zsh" || $2 == "bash" || $2 == "nu" { print $1; exit }'
                )"
                if ! ps -t "$PY_WATCH_TTY" -o comm= 2>/dev/null |
                  grep -qx -e nvim -e codex
                then
                  PY_WATCH_TERMINAL_FREE=1
                fi
              fi
              PY_WATCH_PID="$(cat "$PY_WATCH_PID_FILE" 2>/dev/null || true)"
              if [ "$PY_WATCH_TERMINAL_FREE" = 1 ] && [ -w "$PY_WATCH_TTY_DEVICE" ] &&
                { [ -z "$PY_WATCH_PID" ] || ! kill -0 "$PY_WATCH_PID" 2>/dev/null; }
              then
                mkdir -p "$REPO_ROOT/.direnv"
                PY_WATCH_LOG="$REPO_ROOT/.direnv/python-watcher.$PY_WATCH_TTY_SLUG.log"
                TMPDIR="''${XDG_RUNTIME_DIR:-/tmp}" setsid -f bash -c '
                  for fd in /proc/$$/fd/*; do
                    fd="''${fd##*/}"
                    if [ "$fd" -gt 2 ]; then
                      eval "exec $fd>&-"
                    fi
                  done
                  echo $$ > "$1"
                  pid_file=$1
                  watch_tty=$2
                  tty_device=$3
                  shell_pid=$4
                  shift 4
                  cleanup() {
                    trap - EXIT TERM INT HUP
                    kill "$watcher" 2>/dev/null || true
                    wait "$watcher" 2>/dev/null || true
                    unlink "$pid_file" 2>/dev/null || true
                    if [ -n "$shell_pid" ]; then
                      kill -WINCH "$shell_pid" 2>/dev/null || true
                    fi
                  }
                  trap cleanup EXIT TERM INT HUP
                  "$@" &
                  watcher=$!
                  wait "$watcher" || true
                ' bash "$PY_WATCH_PID_FILE" "$PY_WATCH_TTY" "$PY_WATCH_TTY_DEVICE" "$PY_WATCH_SHELL_PID" \
                  watchexec --restart --quiet --postpone --exts py \
                    --fs-events create,modify,rename \
                    --emit-events-to file \
                    --watch "$REPO_ROOT" \
                    --workdir "$REPO_ROOT" \
                    --shell none \
                    -- bash -c '
                      watch_tty=$1
                      tty_device=$2
                      shell_pid=$3
                      shift 3
                      if [ -z "$watch_tty" ] || [ ! -w "$tty_device" ]; then
                        exit 0
                      fi
                      if ps -t "$watch_tty" -o comm= 2>/dev/null |
                        grep -qx -e nvim -e codex
                      then
                        exit 0
                      fi
                      exec </dev/null >"$tty_device" 2>&1
                      events_file="''${WATCHEXEC_EVENTS_FILE:-}"
                      if [ -z "$events_file" ]; then
                        printf "watcher event had no file list; skipping\n" >&2
                        exit 0
                      fi
                      declare -A seen=()
                      ran=0
                      while IFS= read -r event; do
                        case "$event" in
                          remove:*) continue ;;
                          *:*) file="''${event#*:}" ;;
                          *) continue ;;
                        esac
                        case "$file" in
                          *.py) ;;
                          *) continue ;;
                        esac
                        [ -f "$file" ] || continue
                        [ -z "''${seen[$file]+x}" ] || continue
                        seen["$file"]=1
                        printf "\n[Running: %s]\n" "$file"
                        (
                          cd -- "$(dirname -- "$file")" || exit 1
                          python "$(basename -- "$file")"
                        )
                        ran=1
                      done < "$events_file"
                      if [ "$ran" = 1 ] && [ -n "$shell_pid" ]; then
                        kill -WINCH "$shell_pid" 2>/dev/null || true
                      fi
                    ' watcher-shell "$PY_WATCH_TTY" "$PY_WATCH_TTY_DEVICE" "$PY_WATCH_SHELL_PID" \
                    </dev/null >/dev/null 2>>"$PY_WATCH_LOG"
              fi
              unset PY_WATCH_PID_FILE PY_WATCH_PID PY_WATCH_TTY PY_WATCH_TTY_DEVICE
              unset PY_WATCH_TTY_SLUG PY_WATCH_SHELL_PID PY_WATCH_TERMINAL_FREE PY_WATCH_LOG
            '';
          };
        }
      );

      packages = forAllSystems (system: {
        default = pythonSets.${system}.mkVirtualEnv "hello-world-env" workspace.deps.default;
      });
    };
}
