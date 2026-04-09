import os
import subprocess

# ====== 1️⃣ 设置根目录 ======
AD_DIR = "/your/path/to/data"   # ← 改成你的路径
CN_DIR = "/your/path/to/data"
LMCI_DIR = "/your/path/to/data"

AD_OUTPUT_DIR = "/your/path/to/output"
CN_OUTPUT_DIR = "/your/path/to/output"
LMCI_OUTPUT_DIR = "/your/path/to/output"

# ====== 2️⃣ 是否大小写敏感 ======
IGNORE_CASE = True


def is_mask_file(filename):
    if IGNORE_CASE:
        return "mask" in filename.lower()
    else:
        return "Mask" in filename


def process_file(filepath, flag: int):
    """
    在这里写你要执行的命令
    """
    print(f"Processing: {filepath}")

    # ====== 3️⃣ 在这里写你的命令 ======
    # 示例（你可以改）：
    # cmd = ["python", "your_script.py", "--input", filepath]
    cmd = []
    # 先给你一个占位：
    cmd = ["echo", f"TODO process {filepath}"]
    if flag == 0:
        cmd = ["hd"]
    elif flag == 1:
        cmd = ["nii"]
    elif flag == 2:
        cmd = ["nii.gz"]
    else:
        cmd = ['']

    # 执行命令
    subprocess.run(cmd)


def AD():
    for root, dirs, files in os.walk(AD_DIR):
        for file in files:
            if file.endswith(".nii.gz"):
                if is_mask_file(file):
                    continue

                full_path = os.path.join(root, file)
                process_file(full_path, 2)

def CN():
    for root, dirs, files in os.walk(CN_DIR):
        for file in files:
            if file.endswith(".nii.gz"):
                if is_mask_file(file):
                    continue

                full_path = os.path.join(root, file)
                process_file(full_path, 0)

def LMCI():
    for root, dirs, files in os.walk(LMCI_DIR):
        for file in files:
            if file.endswith(".nii.gz"):
                if is_mask_file(file):
                    continue

                full_path = os.path.join(root, file)
                process_file(full_path,1)

if __name__ == "__main__":
    AD()
    CN()
    LMCI()