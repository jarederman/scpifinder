#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCPI指令搜索Web应用
基于Flask的Web界面，提供简洁的搜索体验
"""

from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

class SCPISearchWeb:
    def __init__(self, json_file_path: str = "scpi_data.json"):
        """初始化SCPI搜索器"""
        self.json_file_path = json_file_path
        self.commands_data = []
        self.command_vectors = []
        self.model_loaded = False
        
        # 加载数据
        self.load_commands_data()
        
        # 延迟加载模型（在第一次搜索时加载）
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        """延迟加载transformer模型"""
        if not self.model_loaded:
            print("正在加载transformer模型...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                self.model_loaded = True
                print("模型加载成功！")
            except Exception as e:
                print(f"模型加载失败，使用备用模型: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                self.model = AutoModel.from_pretrained("distilbert-base-uncased")
                self.model_loaded = True
    
    def load_commands_data(self):
        """加载SCPI指令数据"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.commands_data = json.load(f)
            
            # 提取预计算的向量
            vectors = []
            for cmd in self.commands_data:
                if '向量' in cmd and isinstance(cmd['向量'], list) and len(cmd['向量']) == 384:
                    vector = np.array(cmd['向量'], dtype=np.float32)
                    norm = np.linalg.norm(vector)
                    if abs(norm - 1.0) < 1e-6:
                        vectors.append(vector)
                    else:
                        vectors.append(vector / norm if norm > 1e-8 else np.zeros(384, dtype=np.float32))
                else:
                    vectors.append(np.zeros(384, dtype=np.float32))
            
            self.command_vectors = np.array(vectors, dtype=np.float32)
            print(f"成功加载 {len(self.commands_data)} 条SCPI指令")
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise e
    
    def encode_text(self, text: str) -> np.ndarray:
        """将文本编码为归一化向量"""
        if not self.model_loaded:
            self.load_model()
        
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, 
                              truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            else:
                embeddings = outputs.pooler_output.squeeze().numpy()
        
        embeddings = embeddings.astype(np.float32)
        
        # 处理维度
        if embeddings.shape[0] != 384:
            if embeddings.shape[0] > 384:
                embeddings = embeddings[:384]
            else:
                padded = np.zeros(384, dtype=np.float32)
                padded[:embeddings.shape[0]] = embeddings
                embeddings = padded
        
        # 归一化
        norm = np.linalg.norm(embeddings)
        if norm > 1e-8:
            embeddings = embeddings / norm
        else:
            embeddings = np.random.randn(384).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings)
        
        return embeddings
    
    def search(self, query: str, top_k: int = 5) -> list:
        """执行搜索并返回结果"""
        # 生成查询向量
        query_vector = self.encode_text(query)
        
        # 计算余弦相似度（对归一化向量最有效）
        cos_sim = cosine_similarity([query_vector], self.command_vectors)[0]
        
        # 计算角度相似度
        cos_sim_clipped = np.clip(cos_sim, -1, 1)
        angles = np.arccos(np.abs(cos_sim_clipped))
        angle_sim = 1 - (angles / np.pi)
        
        # 计算欧氏距离相似度
        eucl_dist_squared = 2 * (1 - cos_sim)
        eucl_dist = np.sqrt(np.maximum(eucl_dist_squared, 0))
        eucl_sim = np.exp(-eucl_dist / 2)
        
        # 融合相似度（简化版本，注重余弦相似度）
        fused_scores = 0.6 * cos_sim + 0.3 * angle_sim + 0.1 * eucl_sim
        
        # 获取top-k结果
        top_indices = np.argsort(fused_scores)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            result = {
                'rank': i + 1,
                'score': float(fused_scores[idx]),
                'instruction': self.commands_data[idx]['指令'],
                'description': self.commands_data[idx]['描述'],
                'original_text': self.commands_data[idx]['原文'],
                'cosine_similarity': float(cos_sim[idx])
            }
            results.append(result)
        
        return results

# 创建全局搜索器实例
searcher = SCPISearchWeb()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """搜索API接口"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': '请输入搜索内容'}), 400
        
        # 执行搜索
        results = searcher.search(query, top_k=5)
        
        return jsonify({
            'query': query,
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'搜索失败: {str(e)}'}), 500

@app.route('/health')
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model_loaded': searcher.model_loaded,
        'data_count': len(searcher.commands_data)
    })

if __name__ == '__main__':
    print("正在启动SCPI指令搜索Web应用...")
    print("应用将在 http://localhost:5001 启动")
    app.run(debug=True, host='0.0.0.0', port=5001) 