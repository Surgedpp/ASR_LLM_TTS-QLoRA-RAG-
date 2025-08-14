#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from vehicle_vector_search import VehicleVectorSearch


def print_sections(searcher):
    stats = searcher.get_statistics()
    print(f"\n共有 {stats['total_sections']} 个章节:")
    for i, section in enumerate(stats['sections'], 1):
        print(f"  {i}. {section}")

def interactive_search(model_path):    
    try:
        print("正在加载向量数据库...")
        searcher = VehicleVectorSearch(model_path)
        print("向量数据库加载成功！")
        
        # 显示统计信息
        stats = searcher.get_statistics()
        print(f"\n数据库信息:")
        print(f"  - 总文档数: {stats['total_documents']}")
        print(f"  - 向量维度: {stats['embedding_dimension']}")
        print(f"  - 章节数: {stats['total_sections']}")
            
        while True:
            try:
                user_input = input("\n请输入查询 (输入 'help' 查看帮助): ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                command = parts[0].lower()
                
                query = user_input
                print(f"\n搜索: '{query}'")
                
                results = searcher.search(query, top_k=5)
                searcher.print_search_results(results, query)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"发生错误: {e}")
                
    except Exception as e:
        print(f"初始化失败: {e}")
        print("请确保已运行 vehicle_data_processor.py 生成向量数据库")
        return 1
    
    return 0
