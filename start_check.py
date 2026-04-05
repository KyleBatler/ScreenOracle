from pathlib import Path

project_name = "ScreenOracle"
project_root = Path(".")
data_folder = project_root / "data"

print(f"Проект: {project_name}")
print(f"Абсолютный путь корня: {project_root.resolve()}")
print(f"Путь к папке data: {data_folder.resolve()}")

data_folder.mkdir(exist_ok=True)

subfolders = ["raw", "processed", "interim"]

for folder_name in subfolders:
    folder_path = data_folder / folder_name
    folder_path.mkdir(exist_ok=True)
    print(f"Подпапка готова: {folder_path.resolve()}")

print("\nПроверка структуры: ")
print(f"data -> exists: {data_folder.exists()}, is_dir: {data_folder.is_dir()}")

for folder_name in subfolders:
    folder_path = data_folder / folder_name
    print(
        f"  {folder_name} -> exists: {folder_path.exists()}, is_dir: {folder_path.is_dir()}"
    )

imdb_folder = data_folder / "raw" / "imdb"
imdb_folder.mkdir(parents=True, exist_ok=True)

required_imdb_files = [
    "title.basics.tsv.gz",
    "title.ratings.tsv.gz",
]

print("\nПроверка IMDb-файлов для MVP:")
for file_name in required_imdb_files:
    file_path = imdb_folder / file_name
    file_exists = file_path.exists()
    is_real_file = file_path.is_file()
    size_bytes = file_path.stat().st_size if is_real_file else 0
    is_not_empty = size_bytes > 0
    print(
        f"  {file_name} -> exists: {file_exists}, is_file: {is_real_file}, "
        f"size_bytes: {size_bytes}, not_empty: {is_not_empty}"
    )

print("\nСтруктура папок и проверка IMDb-файлов выполнены.")