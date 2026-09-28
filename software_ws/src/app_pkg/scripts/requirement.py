import sys
import subprocess

venv_path = "/home/admind/software_ws/src/app_pkg/vens/venv1"
output_path = "/home/admind/software_ws/src/app_pkg/vens/venv1/requirements.txt"

# 调用虚拟环境的 pip
cmd = f"{venv_path}/bin/pip freeze"
result = subprocess.check_output(cmd, shell=True, text=True)

# 写入指定文件
with open(output_path, 'w') as f:
    f.write(result)

print(f"Requirements 已导出到 {output_path}")