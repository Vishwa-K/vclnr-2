from pathlib import Path

def print_structure(path: Path, prefix=""):
    for item in sorted(path.iterdir()):
        if item.is_dir():
            print(f"{prefix}📁 {item.name}/")
            print_structure(item, prefix + "    ")
        else:
            print(f"{prefix}📄 {item.name}")

# Start from the current script location
base_path = Path(__file__).resolve().parent
print(f"Project Root: {base_path}")
print_structure(base_path)
