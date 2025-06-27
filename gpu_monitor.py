import time
import socket
import signal
import sys
import requests
import GPUtil


INTERVAL = 2


def get_gpu_usage_gputil():
    try:
        gpus = GPUtil.getGPUs()
        gpu_usage = []
        for gpu in gpus:
            gpu_usage.append(
                {
                    "gpu_id": gpu.id,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "util": gpu.load * 100,
                }
            )
        return gpu_usage
    except Exception as e:
        return f"Error using GPUtil: {e}"


def send_message(message: str):
    bot_token = "7791912773:AAFqy-7ZRwlgIFr8NyPDFTEqa6NurfZpNUQ"
    chat_id = "1800469436"
    assert message is not None
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    proxies = {"http": "http://127.0.0.1:6154", "https": "http://127.0.0.1:6154"}
    response = requests.post(url, json=payload, proxies=proxies)
    if not response.ok:
        print("Send failed:", response.text)


def find_idle_gpu():
    usage = get_gpu_usage_gputil()
    for gpu in usage:
        if gpu["memory_used"] < 1024 and gpu["util"] < 5:
            yield gpu


def signal_handler(sig, frame):
    print("\nCtrl+C entered, exiting...")
    sys.exit(0)


def main():
    hostname = socket.gethostname()
    while True:
        for gpu in find_idle_gpu():
            memory_used = gpu["memory_used"] / 1024
            memory_total = gpu["memory_total"] / 1024
            message = f"Host: {hostname}\nGPU: {gpu['gpu_id']}\nMemory Used: {memory_used:.2f} GiB / {memory_total:.2f} GiB\nGPU Utilization: {gpu['util']:.0f}%"
            send_message(message)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
