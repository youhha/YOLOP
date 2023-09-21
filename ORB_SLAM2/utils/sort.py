import os

# 指定图片所在的文件夹路径
folder_path = '/home/nio/文档/ORB_SLAM2/dataset/kitti/00/image_0'



# 获取文件夹中的所有文件名
file_names = os.listdir(folder_path)

# 过滤出扩展名为.png的文件并排序
image_files = sorted([file for file in file_names if file.endswith('.png')])

# 计算新的文件名前缀的数字部分起始值
# 从000000开始
start_index = 0

# 重新命名图片文件
for i, file in enumerate(image_files, start=start_index):
    # 从文件名中提取扩展名
    _, extension = os.path.splitext(file)
    
    # 构建新的文件名
    new_name = f'{i:06d}{extension}'  # 使用6位数字编号，例如：000000.png
    
    old_path = os.path.join(folder_path, file)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f'Renamed: {file} -> {new_name}')
