import os
import shutil

# ================= 1. 配置你的路径 =================
# 注意：请确保相对路径正确，以下为示例
ORIGINAL_DATA_DIR = './garbage265'           # 原始数据集的目录
NEW_DATA_DIR = './garbage_4_classes'         # 合并后的新目录
CLASSNAME_FILE = './garbage265/classname.txt' # txt 文件的位置
# ===================================================

def build_mapping():
    """让 Python 自动读取 txt 文件并建立映射字典"""
    mapping = {}
    
    # 打开并读取文件
    with open(CLASSNAME_FILE, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        
    for idx, line in enumerate(lines):
        # 过滤掉空行
        if '-' not in line:
            continue
            
        # 用 '-' 把字符串一分为二：前面的做大类，后面的做小类
        big_class, small_class = line.split('-', 1)
        
        # 核心：修复原数据集中的错别字 Bug
        if big_class == '其它垃圾':
            big_class = '其他垃圾'
            
        # 为了绝对安全，我们把三种可能出现的文件夹名字全存进字典里：
        mapping[str(idx)] = big_class       # 应对文件夹名为数字 ID (如 '0') 的情况
        mapping[line] = big_class           # 应对文件夹名为完整名称 (如 '厨余垃圾-八宝粥') 的情况
        mapping[small_class] = big_class    # 应对文件夹名为小类名称 (如 '八宝粥') 的情况
        
    return mapping

def merge_dataset():
    print("正在解析类别映射文件...")
    class_mapping = build_mapping()
    
    # 同时处理训练集和验证集
    for split in ['train', 'val']:
        original_split_dir = os.path.join(ORIGINAL_DATA_DIR, split)
        new_split_dir = os.path.join(NEW_DATA_DIR, split)
        
        if not os.path.exists(original_split_dir):
            print(f"⚠️ 找不到文件夹: {original_split_dir}，跳过处理。")
            continue
            
        print(f"\n🚀 开始合并 [{split}] 集合...")
        
        # 遍历原始的 265 个文件夹
        for sub_class in os.listdir(original_split_dir):
            sub_class_path = os.path.join(original_split_dir, sub_class)
            
            # 忽略隐藏文件或说明文件，确保只处理文件夹
            if not os.path.isdir(sub_class_path):
                continue
                
            # 核心：通过字典查询这个小类属于哪个大类
            target_big_class = class_mapping.get(sub_class, "未知分类")
            
            target_dir = os.path.join(new_split_dir, target_big_class)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                
            # 获取该小类文件夹下的所有图片并开始搬运
            images = os.listdir(sub_class_path)
            for img in images:
                src_img_path = os.path.join(sub_class_path, img)
                
                # 给新图片加个前缀，防止不同小类有同名图片（比如分别叫 001.jpg 导致互相覆盖）
                new_img_name = f"{sub_class}_{img}"
                dst_img_path = os.path.join(target_dir, new_img_name)
                
                # 执行复制操作
                shutil.copy(src_img_path, dst_img_path)
                
        print(f"✅ [{split}] 集合四大类合并完成！")

if __name__ == '__main__':
    merge_dataset()