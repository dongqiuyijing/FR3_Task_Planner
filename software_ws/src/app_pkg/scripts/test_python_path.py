#!/home/admind/software_ws/src/app_pkg/vens/venv2/bin/python
import sys
from pathlib import Path
sys.path.insert(0,"/home/admind/software_ws/devel/lib/python3/dist-packages")

def analyze_paths():
    print("\n=== Python路径分析 ===")
    path_types = {
        "标准库": [],
        "第三方包": [],
        "项目路径": [],
        "其他": []
    }

    for path in sys.path:
        path = str(Path(path).resolve())  # 标准化路径
        if not path:
            continue
            
        if 'site-packages' in path:
            path_types["第三方包"].append(path)
        elif 'lib/python' in path:
            path_types["标准库"].append(path)
        elif Path(path).name in ('src', 'scripts') or Path.cwd() in Path(path).parents:
            path_types["项目路径"].append(path)
        else:
            path_types["其他"].append(path)

    # 打印分类结果
    for category, paths in path_types.items():
        print(f"\n◆ {category} ({len(paths)}条):")
        for i, path in enumerate(paths, 1):
            print(f"    {i}. {path}")

analyze_paths()
print(f"\n当前解释器: {sys.executable}")
# 测试代码
import numpy
print(f"=== VENV1 环境 ===")
print(f"Python解释器: {sys.executable}")
print(f"NumPy版本: {numpy.__version__}")
print(f"NumPy路径: {numpy.__file__}")