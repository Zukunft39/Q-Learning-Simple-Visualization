"""运行 python BuildExe.py，在 dist 中生成单文件桌面程序。"""
import os
import sys
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent
    local_tools = folder.parents[1] / ".build_tools"
    if local_tools.exists():
        sys.path.insert(0, str(local_tools))
    try:
        import PyInstaller.__main__
    except ImportError:
        raise SystemExit("请先安装 PyInstaller：python -m pip install pyinstaller")

    os.environ["PYINSTALLER_CONFIG_DIR"] = str(folder / "build" / "cache")
    os.environ["MPLCONFIGDIR"] = str(folder / "build" / "matplotlib")
    options = [
        str(folder / "LearningLab.py"),
        "--name", "GridWorldLearningLab",
        "--onefile", "--windowed", "--noconfirm", "--noupx",
        "--distpath", str(folder / "dist"),
        "--workpath", str(folder / "build" / "pyinstaller"),
        "--specpath", str(folder / "build"),
        "--paths", str(folder),
        "--hidden-import", "matplotlib.backends.backend_tkagg",
    ]
    # 本程序只用 Tk，避免把本机其他大型可选库也打进 EXE。
    for name in ("IPython", "pytest", "PyQt5", "PyQt6", "PySide2",
                 "PySide6", "scipy", "pandas", "torch", "cv2"):
        options.extend(["--exclude-module", name])
    PyInstaller.__main__.run(options)


if __name__ == "__main__":
    main()
