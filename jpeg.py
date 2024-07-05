import os
import shutil


def rename_and_copy_jpeg_files(src_directory, dst_directory):
    # 目的ディレクトリが存在しない場合は作成
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)

    for filename in os.listdir(src_directory):
        if filename.lower().endswith(".jpeg"):
            new_filename = filename.rsplit(".", 1)[0] + ".jpeg"
            src_file = os.path.join(src_directory, filename)
            dst_file = os.path.join(dst_directory, new_filename)
            shutil.copy2(src_file, dst_file)
            print(f"Copied and renamed: {src_file} -> {dst_file}")


# 使用例
src_directory_path = "/path/to/your/source_directory"
dst_directory_path = "/path/to/your/destination_directory"
rename_and_copy_jpeg_files(src_directory_path, dst_directory_path)
