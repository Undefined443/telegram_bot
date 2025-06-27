import signal
import sys
import time
import torch as th


MEM_FREE = 0.1
LATENCY = 0.08
#LATENCY = 1


def signal_handler(sig, frame):
    print("\nCtrl+C entered, exiting...")
    sys.exit(0)


def main():
    tensors = []
    models = []
    num_gpus = th.cuda.device_count()
    for i in range(num_gpus):
        device = th.device(f"cuda:{i}")
        free_memory, _ = th.cuda.mem_get_info(device=device)
        margin = int(free_memory * MEM_FREE)
        unit_size = 2**30
        num_units = (free_memory - margin) // unit_size
        num_elems = unit_size // 4
        if num_units <= 0:
            print(f"No enough memory on GPU {i}, skip")
            continue

        try:
            tensor = th.empty(num_units, num_elems, dtype=th.float32, device=device)
            tensors.append(tensor)
            mem_size = num_units * unit_size // 2**30
            print(f"Successfully preserved {mem_size}GB of memory on GPU {i}.")
            model = th.nn.Linear(num_elems, 1, device=device)
            models.append(model)
        except Exception as e:
            print(f"Failed to preserve memory on GPU {i}: {e}")

    while True:
        for i, tensor in enumerate(tensors):
            model = models[i]
            _ = model(tensor)
        time.sleep(LATENCY)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
