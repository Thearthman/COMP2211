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

      overlay = lib.composeManyExtensions [
        (workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        })
        (final: prev:
          # NVIDIA wheels dlopen their sibling libraries and the host driver at
          # runtime; LD_LIBRARY_PATH is populated by the dev-shell hook below.
          let
            nvidiaPackages = builtins.filter
              (name: lib.hasPrefix "nvidia-" name)
              (builtins.attrNames prev);
          in
          builtins.listToAttrs (map
            (name: {
              inherit name;
              value = prev.${name}.overrideAttrs (_: {
                autoPatchelfIgnoreMissingDeps = true;
              });
            })
            nvidiaPackages))
      ];

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
              pkgs.basedpyright
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

              # TensorFlow dlopen()s CUDA/cuDNN instead of linking them, so the
              # NVIDIA wheel libraries and the host driver must be discoverable.
              _tf_site_packages="$(
                python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
              )"
              _tf_cuda_libs=""
              while IFS= read -r -d "" _lib_dir; do
                _tf_cuda_libs="''${_tf_cuda_libs:+$_tf_cuda_libs:}$_lib_dir"
              done < <(find -L "$_tf_site_packages/nvidia" -type d -name lib -print0 2>/dev/null)
              _driver_lib="$(
                find /run/opengl-driver/lib /usr/lib /lib \
                  -name libcuda.so.1 -print -quit 2>/dev/null
              )"
              if [ -n "$_driver_lib" ]; then
                _driver_dir="$REPO_ROOT/.direnv/nvidia-driver"
                mkdir -p "$_driver_dir"
                for _name in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
                  _src="$(dirname "$_driver_lib")/$_name"
                  [ -e "$_src" ] && ln -sfn "$_src" "$_driver_dir/$_name"
                done
                _tf_cuda_libs="$_driver_dir''${_tf_cuda_libs:+:$_tf_cuda_libs}"
              fi
              if [ -n "$_tf_cuda_libs" ]; then
                export LD_LIBRARY_PATH="$_tf_cuda_libs''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              fi
              # TensorFlow only ships PTX for sm_90; let XLA use the CUDA
              # package's matching ptxas/libdevice when JIT-compiling for sm_120.
              export XLA_FLAGS="--xla_gpu_cuda_data_dir=$_tf_site_packages/nvidia/cuda_nvcc''${XLA_FLAGS:+ $XLA_FLAGS}"
              unset _tf_site_packages _tf_cuda_libs _lib_dir _driver_lib _driver_dir _name _src

              PY_WATCH_TTY="$(ps -o tty= -p $$ 2>/dev/null | tr -d '[:space:]')"
              PY_WATCH_TTY_DEVICE=""
              PY_WATCH_PID_FILE=""
              PY_WATCH_ENV_FILE=""
              PY_WATCH_SHELL_PID=""
              PY_WATCH_TERMINAL_FREE=0
              if [ -n "$PY_WATCH_TTY" ] && [ "$PY_WATCH_TTY" != "?" ]; then
                PY_WATCH_TTY_DEVICE="/dev/$PY_WATCH_TTY"
                PY_WATCH_TTY_SLUG="''${PY_WATCH_TTY//\//_}"
                PY_WATCH_PID_FILE="$REPO_ROOT/.direnv/python-watcher.$PY_WATCH_TTY_SLUG.pid"
                PY_WATCH_ENV_FILE="$PY_WATCH_PID_FILE.env"
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
              case "$PY_WATCH_PID" in
                "" | *[!0-9]*) PY_WATCH_PID="" ;;
              esac
              PY_WATCH_EXPECTED_ENV=""
              if [ -n "$PY_WATCH_ENV_FILE" ]; then
                PY_WATCH_EXPECTED_ENV="$(
                  printf 'LD_LIBRARY_PATH=%s\nXLA_FLAGS=%s\n' \
                    "$LD_LIBRARY_PATH" "$XLA_FLAGS"
                )"
              fi
              PY_WATCH_STALE=0
              if [ -n "$PY_WATCH_PID" ] && kill -0 "$PY_WATCH_PID" 2>/dev/null; then
                if ! tr '\0' ' ' < "/proc/$PY_WATCH_PID/cmdline" 2>/dev/null |
                  grep -Fq "$PY_WATCH_PID_FILE"
                then
                  # The PID was reused by an unrelated process.
                  PY_WATCH_PID=""
                  unlink "$PY_WATCH_PID_FILE" 2>/dev/null || true
                  unlink "$PY_WATCH_ENV_FILE" 2>/dev/null || true
                else
                  # A watcher inherited the shell environment when it started.
                  # Reusing one after a flake/env reload would run Python without
                  # the current NVIDIA wheel paths, so replace it when stale.
                  PY_WATCH_LIVE_ENV="$(cat "$PY_WATCH_ENV_FILE" 2>/dev/null || true)"
                  if [ "$PY_WATCH_LIVE_ENV" != "$PY_WATCH_EXPECTED_ENV" ]; then
                    PY_WATCH_STALE=1
                  fi
                fi
              fi
              if [ "$PY_WATCH_TERMINAL_FREE" = 1 ] && [ -w "$PY_WATCH_TTY_DEVICE" ] &&
                { [ -z "$PY_WATCH_PID" ] ||
                  ! kill -0 "$PY_WATCH_PID" 2>/dev/null ||
                  [ "$PY_WATCH_STALE" = 1 ]; }
              then
                mkdir -p "$REPO_ROOT/.direnv"
                PY_WATCH_LOG="$REPO_ROOT/.direnv/python-watcher.$PY_WATCH_TTY_SLUG.log"
                if [ -n "$PY_WATCH_PID" ] && kill -0 "$PY_WATCH_PID" 2>/dev/null; then
                  kill "$PY_WATCH_PID" 2>/dev/null || true
                  for _ in 1 2 3 4 5 6 7 8 9 10; do
                    kill -0 "$PY_WATCH_PID" 2>/dev/null || break
                    sleep 0.1
                  done
                  kill -KILL "$PY_WATCH_PID" 2>/dev/null || true
                  unlink "$PY_WATCH_PID_FILE" 2>/dev/null || true
                  unlink "$PY_WATCH_ENV_FILE" 2>/dev/null || true
                fi
                printf '%s\n' "$PY_WATCH_EXPECTED_ENV" > "$PY_WATCH_ENV_FILE"
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
                    unlink "$pid_file.env" 2>/dev/null || true
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
              unset PY_WATCH_PID_FILE PY_WATCH_ENV_FILE PY_WATCH_PID PY_WATCH_TTY PY_WATCH_TTY_DEVICE
              unset PY_WATCH_TTY_SLUG PY_WATCH_SHELL_PID PY_WATCH_TERMINAL_FREE
              unset PY_WATCH_STALE PY_WATCH_LIVE_ENV PY_WATCH_EXPECTED_ENV PY_WATCH_LOG
            '';
          };
        }
      );

      packages = forAllSystems (system: {
        default = pythonSets.${system}.mkVirtualEnv "hello-world-env" workspace.deps.default;
      });
    };
}
