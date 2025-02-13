import subprocess
import time
import os

def get_gpu_utilization():
    # 使用 nvidia-smi 命令获取 GPU 使用情况
    result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        gpu_usage = int(result.stdout.strip().split('\n')[0])
        return gpu_usage
    else:
        print("无法获取 GPU 使用率：", result.stderr)
        return None

def check_and_shutdown(threshold=10, check_interval=30, max_low_usage_checks=3):
    low_usage_count = 0

    while True:
        gpu_usage = get_gpu_utilization()

        if gpu_usage is not None:
            print(f"当前 GPU 使用率: {gpu_usage}%")

            if gpu_usage < threshold:
                low_usage_count += 1
                print(f"低于 {threshold}% 的次数：{low_usage_count}/{max_low_usage_checks}")
            else:
                low_usage_count = 0  # 重置计数

            # 检查是否连续达到低占用阈值
            if low_usage_count >= max_low_usage_checks:
                # zip压缩当前目录
                try:
                    parent_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
                    print(f"当前父目录：{parent_path}")
                    os.system(f"zip -r {parent_path}.zip {parent_path}")
                except Exception as e:
                    print(f"压缩失败：{e}")

                print("GPU 使用率持续过低，关闭服务器...")
                os.system("/usr/bin/shutdown")  # 执行关机命令
                break
        else:
            print("无法检测到 GPU 使用情况，将稍后重试。")
            "../DQN_SV_RS_new_2"

        time.sleep(check_interval)


if __name__ == '__main__':
    check_and_shutdown()
