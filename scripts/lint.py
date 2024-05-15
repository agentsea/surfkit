import subprocess


def main():
    subprocess.run(["black", "."])
    subprocess.run(["isort", "."])


if __name__ == "__main__":
    main()
