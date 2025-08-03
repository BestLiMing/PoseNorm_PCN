import os
import sys
from pathlib import Path


def get_project_root(markers=None) -> Path:
    DEFAULT_MARKERS = [
        '.projectroot',
        'requirements.txt',
        'README.md',
    ]
    check_markers = markers or DEFAULT_MARKERS
    if getattr(sys, 'frozen', False):
        start_path = Path(sys.executable).parent
    else:
        start_path = Path(__file__).parent
    current_path = start_path.resolve()
    while True:
        if any((current_path / marker).exists() for marker in check_markers):
            return current_path
        if current_path.parent == current_path:
            break

        current_path = current_path.parent
    raise FileNotFoundError(
        f"Project root directory not found! Please perform the following operations：\n"
        f"1. Create a marker file in the root directory.：touch {Path(os.path.dirname(os.path.dirname(__file__))) / '.projectroot'}\n"
        f"2. Or modify the identification mark parameters"
    )


if __name__ == "__main__":
    try:
        root = get_project_root()
        print(f"✅ 正确识别根目录：{root}")
        print(f"→ 是否符合预期：{root == Path(os.path.dirname(os.path.dirname(__file__)))}")
    except Exception as e:
        print(f"❌ 错误：{str(e)}")
