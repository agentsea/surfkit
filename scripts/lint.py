import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Run linters.")
    parser.add_argument("--check", action="store_true", help="Run in check mode.")
    args = parser.parse_args()

    black_command = ["black", "."]
    isort_command = ["isort", "."]

    if args.check:
        black_command = ["black", "--check", "--diff", "."]
        isort_command = ["isort", "--check", "--diff", "."]

    subprocess.run(black_command, check=True)
    # subprocess.run(isort_command, check=True)


if __name__ == "__main__":
    main()
