#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import sentence_transformers
import numpy as np
import sklearn

from vehicle_data_processor import VehicleDataProcessor
from vehicle_vector_search import VehicleVectorSearch


def print_demo_banner():
    print("=" * 80)
    print("车载文本向量化系统演示")
    print("基于 text2vec-base-chinese 模型")
    print("=" * 80)
    print()



def demo_data_processing(model_path):
    print("\n数据处理和向量化")
    print("-" * 50)
    
    input_file = "vehicle_manual_data.txt"
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return False
    
    try:
        processor = VehicleDataProcessor(os.path.normpath(model_path))
        
        start_time = time.time()
        processor.process_vehicle_data(input_file)
        end_time = time.time()
        
        print(f"数据处理完成！耗时: {end_time - start_time:.2f} 秒")
        return True
        
    except Exception as e:
        print(f"数据处理失败: {e}")
        return False

def demo_interactive_search(model_path):
    print("\n交互式搜索演示")
    print("-" * 50)
    
    try:
        from interactive_search import interactive_search
        print("启动交互式搜索界面...")
        print("在交互界面中，您可以尝试以下查询:")
        print("  - 发动机故障")
        print("  - 制动系统")
        print("  - 自动泊车")
        print("\n输入 'quit' 退出演示")
        
        interactive_search(model_path)
        return True
        
    except Exception as e:
        print(f"交互式搜索失败: {e}")
        return False

def show_file_structure():
    print("\n生成的文件结构:")
    print("-" * 50)
    
    vector_db_dir = "vector_db"
    if os.path.exists(vector_db_dir):
        for root, dirs, files in os.walk(vector_db_dir):
            level = root.replace(vector_db_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"{subindent}{file} ({file_size} bytes)")
    else:
        print("vector_db 目录不存在")

def main():
    print_demo_banner()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models")
    if not demo_data_processing(model_path):
        print("数据处理失败，无法继续演示")
        return 1
    
    show_file_structure()
    
    print("\n是否启动交互式搜索界面? (y/n): ", end="")
    try:
        user_input = input().strip().lower()
        if user_input in ['y', 'yes', '是']:
            demo_interactive_search(model_path)
    except KeyboardInterrupt:
        print("\n演示结束")
    
    print("\n演示完成！")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 