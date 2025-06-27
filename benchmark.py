import torch as th
import time
import sys
import signal
from tqdm import tqdm


def signal_handler(sig, frame):
    print("\nCtrl+C entered, exiting...")
    sys.exit(0)


def main():
    try:
        benchmark()
    except th.OutOfMemoryError as e:
        print(f"CUDA out of memory.")
        sys.exit(1)


def benchmark():
    unit_size = 2**30  # 1GiB
    num_units = 76
    num_elems = unit_size // 4
    device = th.device("cuda")
    tensor = th.empty(num_units - 1, num_elems, device=device)
    model = th.nn.Linear(num_elems, 1, device=device)

    start = time.time()
    for _ in range(10000):
        pass
    end = time.time()
    loop_time = end - start

    for _ in range(500):
        _ = model(tensor)  # warm up

    start = time.time()
    for _ in tqdm(range(10000)):
        _ = model(tensor)
    end = time.time()
    model_time = end - start - loop_time
    print(f"Time taken: {model_time:.2f} seconds")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
