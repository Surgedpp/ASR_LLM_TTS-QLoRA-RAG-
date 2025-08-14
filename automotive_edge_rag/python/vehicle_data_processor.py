#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VehicleDataProcessor:
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.texts = []
        self.metadata = []
        
        os.makedirs("vector_db", exist_ok=True)
        os.makedirs("processed_data", exist_ok=True)
        
    def load_model(self):
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def load_vehicle_data(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info(f"正在加载数据文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        sections = self._parse_vehicle_manual(content)
        return sections
    
    def _parse_vehicle_manual(self, content: str) -> List[Dict[str, Any]]:
        sections = []
        lines = content.split('\n')
        
        current_section = None
        current_subsection = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('## '):
                if current_section:
                    sections.append({
                        'section': current_section,
                        'subsection': current_subsection,
                        'content': '\n'.join(current_content),
                        'type': 'section'
                    })
                
                current_section = line[3:]
                current_subsection = None
                current_content = []
                
            # 检测子标题 (###)
            elif line.startswith('### '):
                if current_subsection and current_content:
                    sections.append({
                        'section': current_section,
                        'subsection': current_subsection,
                        'content': '\n'.join(current_content),
                        'type': 'subsection'
                    })
                
                current_subsection = line[4:]
                current_content = []
                
            else:
                current_content.append(line)
        
        if current_section and current_content:
            sections.append({
                'section': current_section,
                'subsection': current_subsection,
                'content': '\n'.join(current_content),
                'type': 'subsection' if current_subsection else 'section'
            })
        
        logger.info(f"解析完成，共生成 {len(sections)} 个文本片段")
        return sections
    
    def prepare_texts_for_embedding(self, data: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        texts = []
        metadata = []
        
        for i, item in enumerate(data):
            text_parts = []
            if item['section']:
                text_parts.append(f"章节: {item['section']}")
            if item['subsection']:
                text_parts.append(f"子章节: {item['subsection']}")
            if item['content']:
                text_parts.append(item['content'])
            
            text = " | ".join(text_parts)
            texts.append(text)
            
            meta = {
                'id': i,
                'section': item['section'],
                'subsection': item['subsection'],
                'type': item['type'],
                'content_length': len(item['content'])
            }
            metadata.append(meta)
        
        return texts, metadata
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            self.load_model()
        
        logger.info(f"正在为 {len(texts)} 个文本生成向量...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"向量生成完成，形状: {embeddings.shape}")
        
        return embeddings
    
    def save_vector_database(self, embeddings: np.ndarray, texts: List[str], metadata: List[Dict[str, Any]]):
        # 保存向量数据
        vector_file = "vector_db/vehicle_embeddings.npy"
        np.save(vector_file, embeddings)
        logger.info(f"向量数据已保存到: {vector_file}")
        
        # 保存文本和元数据
        data_file = "vector_db/vehicle_data.pkl"
        data = {
            'texts': texts,
            'metadata': metadata,
            'model_name': self.model_name
        }
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"文本和元数据已保存到: {data_file}")
        
        json_file = "vector_db/vehicle_data.json"
        json_data = {
            'texts': texts,
            'metadata': metadata,
            'model_name': self.model_name,
            'embedding_shape': embeddings.shape
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON格式数据已保存到: {json_file}")
    
    def create_search_index(self, embeddings: np.ndarray, texts: List[str], metadata: List[Dict[str, Any]]):

        # 计算向量相似度矩阵（可选，用于快速检索）
        similarity_matrix = cosine_similarity(embeddings)
        
        # 保存相似度矩阵
        similarity_file = "vector_db/similarity_matrix.npy"
        np.save(similarity_file, similarity_matrix)
        logger.info(f"相似度矩阵已保存到: {similarity_file}")
        
        # 创建索引信息
        index_info = {
            'total_documents': len(texts),
            'embedding_dimension': embeddings.shape[1],
            'model_name': self.model_name,
            'created_at': str(np.datetime64('now'))
        }
        
        index_file = "vector_db/index_info.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_info, f, ensure_ascii=False, indent=2)
        logger.info(f"索引信息已保存到: {index_file}")
    
    def process_vehicle_data(self, input_file: str):
        logger.info("开始处理车载数据...")
        
        # 1. 加载数据
        data = self.load_vehicle_data(input_file)
        
        # 2. 准备文本
        texts, metadata = self.prepare_texts_for_embedding(data)
        
        # 3. 生成向量
        embeddings = self.generate_embeddings(texts)
        
        # 4. 保存向量数据库
        self.save_vector_database(embeddings, texts, metadata)
        
        # 5. 创建搜索索引
        self.create_search_index(embeddings, texts, metadata)
        
        logger.info("车载数据处理完成！")
        
        # 输出统计信息
        logger.info(f"处理统计:")
        logger.info(f"  - 文本片段数量: {len(texts)}")
        logger.info(f"  - 向量维度: {embeddings.shape[1]}")
        logger.info(f"  - 章节数量: {len(set(m['section'] for m in metadata if m['section']))}")
        logger.info(f"  - 子章节数量: {len(set(m['subsection'] for m in metadata if m['subsection']))}")


def main():
    """主函数"""
    # 创建处理器
    processor = VehicleDataProcessor()
    
    # 处理数据
    input_file = "vehicle_manual_data.txt"
    processor.process_vehicle_data(input_file)


if __name__ == "__main__":
    main() 
