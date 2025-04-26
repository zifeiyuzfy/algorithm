import os

txt_file_path = 'D:/YOLO11Traindata/CocoAndVis2/test/labels'  # 原始的标签路径

labels_name = os.listdir(txt_file_path)  # 获得每一个标签名字的列表
for name in labels_name:  # 遍历每一个文件
    file_path = os.path.join(txt_file_path, name)
    # 读取文件内容
    with open(file_path, 'r') as read_file:
        fline = read_file.readlines()

    new_lines = []
    for j in fline:  # 遍历txt文件中每一行
        list1 = j.split()
        if list1[0] == '1':
            list1[0] = '0'  # 将类别1改成类别0
        new_line = " ".join(list1) + '\n'
        new_lines.append(new_line)

    # 将处理后的内容写回原文件
    with open(file_path, 'w') as save_txt:
        save_txt.writelines(new_lines)