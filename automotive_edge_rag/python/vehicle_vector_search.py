#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VehicleVectorSearch:
    """车载向量搜索类"""
    
    def __init__(self, model_path,vector_db_path: str = "vector_db"):
        self.vector_db_path = vector_db_path
        self.model = None
        self.model_path=model_path
        self.embeddings = None
        self.texts = []
        self.metadata = []
        self.similarity_matrix = None
        
        self._load_vector_database()
        
    def _load_vector_database(self):
        try:
            embeddings_file = os.path.join(self.vector_db_path, "vehicle_embeddings.npy")
            if os.path.exists(embeddings_file):
                self.embeddings = np.load(embeddings_file)
                logger.info(f"向量数据加载成功，形状: {self.embeddings.shape}")
            else:
                raise FileNotFoundError(f"向量文件不存在: {embeddings_file}")
            
            # 加载文本和元数据
            data_file = os.path.join(self.vector_db_path, "vehicle_data.pkl")
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                self.texts = data['texts']
                self.metadata = data['metadata']
                logger.info(f"文本和元数据加载成功，共 {len(self.texts)} 个文档")
            else:
                raise FileNotFoundError(f"数据文件不存在: {data_file}")
            
            similarity_file = os.path.join(self.vector_db_path, "similarity_matrix.npy")
            if os.path.exists(similarity_file):
                self.similarity_matrix = np.load(similarity_file)
                logger.info("相似度矩阵加载成功")
            
        except Exception as e:
            logger.error(f"加载向量数据库失败: {e}")
            raise
    
    def load_model(self, model_name):
        try:
            logger.info(f"正在加载模型: {model_name}")
            
            model_paths = [
                model_name
            ]
            
            model_loaded = False
            for path in model_paths:
                try:
                    logger.info(f"尝试加载模型: {path}")
                    self.model = SentenceTransformer(path)
                    logger.info(f"模型加载成功: {path}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"模型加载失败: {path} - {e}")
                    continue
            
            if not model_loaded:
                raise Exception("所有模型路径都无法加载")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        if self.model is None:
            self.load_model(self.model_path)
        
        query_embedding = self.model.encode([query])
        
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                result = {
                    'id': idx,
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(similarity),
                    'section': self.metadata[idx]['section'],
                    'subsection': self.metadata[idx]['subsection']
                }
                results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        sections = set(m['section'] for m in self.metadata if m['section'])
        subsections = set(m['subsection'] for m in self.metadata if m['subsection'])
        
        return {
            'total_documents': len(self.texts),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'total_sections': len(sections),
            'total_subsections': len(subsections),
            'sections': list(sections),
            'has_similarity_matrix': self.similarity_matrix is not None
        }
    
    def print_search_results(self, results: List[Dict[str, Any]], query: str = ""):
        if query:
            print(f"\n查询: '{query}'")
        
        if not results:
            print("未找到相关结果")
            return
        
        print(f"\n找到 {len(results)} 个相关结果:")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 相似度: {result['similarity']:.4f}")
            print(f"   章节: {result['section']}")
            if result['subsection']:
                print(f"   子章节: {result['subsection']}")
            print(f"   内容: {result['text'][:200]}...")
            print("-" * 40)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models")
    searcher = VehicleVectorSearch(os.path.normpath(model_path))
    
    stats = searcher.get_statistics()
    print("向量数据库统计信息:")
    print(f"  - 总文档数: {stats['total_documents']}")
    print(f"  - 向量维度: {stats['embedding_dimension']}")
    print(f"  - 章节数: {stats['total_sections']}")
    print(f"  - 子章节数: {stats['total_subsections']}")
    
    test_queries = [
        "发动机故障",
        "制动系统",
        "保养周期",
        "安全气囊",
        "燃油经济性",
        "天气怎么样"
    ]
    
    print("\n" + "="*80)
    print("搜索演示")
    print("="*80)
    
    for query in test_queries:
        start_time = time.time()
 
        results = searcher.search(query, top_k=1)
        searcher.print_search_results(results, query)
        end_time = time.time()
        
        print(f"数据处理完成！耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main() 
