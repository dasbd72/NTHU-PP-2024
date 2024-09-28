import os

source = "lab2.cc"
targets = ["lab2_hybrid.cc", "lab2_omp.cc", "lab2_pthread.cc"]


def sync():
    if not os.path.exists(source):
        raise Exception("Source file not found")
    with open(source, "r") as f:
        data = f.read()
    for target in targets:
        with open(target, "w") as f:
            f.write(data)
        print(f"Synced {source} to {target}")


if __name__ == "__main__":
    sync()
