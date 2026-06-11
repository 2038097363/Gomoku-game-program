from __future__ import annotations

import subprocess


def main() -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        print("nvidia-smi not found. NVIDIA driver tools may not be installed.")
        return
    except subprocess.CalledProcessError as exc:
        print(exc.stdout)
        print(exc.stderr)
        raise SystemExit(exc.returncode)

    print(result.stdout)


if __name__ == "__main__":
    main()
