#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCPI指令智能搜索工具 - 最终优化版本
专门针对归一化向量优化的搜索算法
确保向量匹配精确性和搜索准确性
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import argparse
import sys
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

class SCPICommandSearcherFinal:
    def __init__(self, json_file_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化最终优化版SCPI指令搜索器
        
        Args:
            json_file_path: SCPI指令JSON文件路径
            model_name: 使用的transformer模型名称
        """
        self.json_file_path = json_file_path
        self.model_name = model_name
        self.commands_data = []
        self.command_vectors = []
        
        # 初始化模型
        print("正在加载transformer模型...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            print(f"模型 {model_name} 加载成功！")
        except Exception as e:
            print(f"模型加载失败，使用备用模型: {e}")
            self.model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
        
        # 加载SCPI指令数据
        self.load_commands_data()
        
    def load_commands_data(self):
        """加载SCPI指令数据并验证向量格式"""
        print("正在加载SCPI指令数据...")
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.commands_data = json.load(f)
            
            # 提取并验证预计算的向量
            vectors = []
            invalid_count = 0
            
            for i, cmd in enumerate(self.commands_data):
                if '向量' in cmd and isinstance(cmd['向量'], list) and len(cmd['向量']) == 384:
                    vector = np.array(cmd['向量'], dtype=np.float32)
                    
                    # 验证向量是否归一化
                    norm = np.linalg.norm(vector)
                    if abs(norm - 1.0) < 1e-6:  # 允许小的数值误差
                        vectors.append(vector)
                    else:
                        print(f"警告: 第{i}个向量未归一化 (模长: {norm:.6f}), 将进行归一化")
                        vectors.append(vector / norm if norm > 1e-8 else np.zeros(384, dtype=np.float32))
                        invalid_count += 1
                else:
                    print(f"警告: 第{i}个条目向量格式异常，使用零向量替代")
                    vectors.append(np.zeros(384, dtype=np.float32))
                    invalid_count += 1
            
            self.command_vectors = np.array(vectors, dtype=np.float32)
            
            print(f"成功加载 {len(self.commands_data)} 条SCPI指令")
            print(f"向量维度: {self.command_vectors.shape}")
            print(f"异常向量数量: {invalid_count}")
            
            # 验证向量归一化
            norms = np.linalg.norm(self.command_vectors, axis=1)
            print(f"向量模长范围: [{norms.min():.6f}, {norms.max():.6f}]")
            print(f"所有向量都已归一化: {np.allclose(norms, 1.0, atol=1e-6)}")
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            sys.exit(1)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        将文本编码为归一化向量，确保与预计算向量格式一致
        
        Args:
            text: 输入文本
            
        Returns:
            归一化的文本向量表示
        """
        # 对文本进行tokenization
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, 
                              truncation=True, max_length=512)
        
        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用平均池化获得句子级表示
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            else:
                embeddings = outputs.pooler_output.squeeze().numpy()
        
        # 确保数据类型一致
        embeddings = embeddings.astype(np.float32)
        
        # 处理维度不匹配的情况
        if embeddings.shape[0] != 384:
            if embeddings.shape[0] > 384:
                embeddings = embeddings[:384]  # 截断
            else:
                # 填充零
                padded = np.zeros(384, dtype=np.float32)
                padded[:embeddings.shape[0]] = embeddings
                embeddings = padded
        
        # 归一化向量以与预计算向量保持一致
        norm = np.linalg.norm(embeddings)
        if norm > 1e-8:
            embeddings = embeddings / norm
        else:
            # 如果向量全零，使用随机单位向量
            embeddings = np.random.randn(384).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings)
        
        return embeddings
    
    def calculate_similarities_optimized(self, query_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """
        针对归一化向量优化的相似度计算
        
        Args:
            query_vector: 查询向量（已归一化）
            
        Returns:
            包含不同相似度计算结果的字典
        """
        similarities = {}
        
        # 1. 余弦相似度 - 对归一化向量最重要
        cos_sim = cosine_similarity([query_vector], self.command_vectors)[0]
        similarities['cosine'] = cos_sim
        
        # 2. 基于余弦相似度的角度距离
        # 对于归一化向量，角度 = arccos(cosine_similarity)
        cos_sim_clipped = np.clip(cos_sim, -1, 1)
        angles = np.arccos(np.abs(cos_sim_clipped))  # 使用绝对值考虑方向
        angle_sim = 1 - (angles / np.pi)  # 归一化到[0,1]，角度越小相似度越高
        similarities['angle'] = angle_sim
        
        # 3. 欧氏距离相似度（对归一化向量优化）
        # 对于单位向量: euclidean_distance^2 = 2 * (1 - cosine_similarity)
        eucl_dist_squared = 2 * (1 - cos_sim)
        eucl_dist = np.sqrt(np.maximum(eucl_dist_squared, 0))  # 避免数值误差导致负数
        eucl_sim = np.exp(-eucl_dist / 2)  # 使用指数衰减转换为相似度
        similarities['euclidean'] = eucl_sim
        
        # 4. 点积相似度（对归一化向量等同于余弦相似度）
        dot_products = np.dot(self.command_vectors, query_vector)
        similarities['dot_product'] = dot_products
        
        # 5. 软余弦相似度（考虑特征重要性）
        # 对于文本向量，某些维度可能更重要
        feature_weights = np.abs(query_vector) + 0.1  # 查询向量的权重
        weighted_cos = np.sum(self.command_vectors * query_vector * feature_weights, axis=1)
        weighted_cos = weighted_cos / np.sum(feature_weights)  # 归一化
        similarities['weighted_cosine'] = weighted_cos
        
        # 6. 基于位置的相似度（考虑向量分量的重要性差异）
        # 为不同位置的特征分配不同权重
        position_weights = np.linspace(1.2, 0.8, 384)  # 前面的特征权重稍高
        position_sim = np.sum(self.command_vectors * query_vector * position_weights, axis=1)
        position_sim = position_sim / np.sum(position_weights)
        similarities['position_weighted'] = position_sim
        
        return similarities
    
    def adaptive_weighted_fusion_final(self, similarities: Dict[str, np.ndarray], 
                                     query_vector: np.ndarray) -> np.ndarray:
        """
        最终优化的自适应权重融合
        
        Args:
            similarities: 不同相似度计算结果
            query_vector: 查询向量
            
        Returns:
            融合后的相似度分数
        """
        # 分析查询向量特征
        query_sparsity = np.sum(np.abs(query_vector) < 0.01) / len(query_vector)
        query_max_component = np.max(np.abs(query_vector))
        query_entropy = -np.sum(np.abs(query_vector) * np.log(np.abs(query_vector) + 1e-8))
        
        # 基础权重配置
        base_weights = {
            'cosine': 0.30,
            'angle': 0.20,
            'euclidean': 0.15,
            'dot_product': 0.15,
            'weighted_cosine': 0.10,
            'position_weighted': 0.10
        }
        
        # 根据查询特征动态调整权重
        if query_sparsity > 0.5:  # 稀疏查询，增加余弦相似度权重
            base_weights['cosine'] += 0.1
            base_weights['weighted_cosine'] += 0.05
            base_weights['euclidean'] -= 0.075
            base_weights['position_weighted'] -= 0.075
        
        if query_max_component > 0.3:  # 有明显主要特征
            base_weights['position_weighted'] += 0.05
            base_weights['angle'] += 0.05
            base_weights['euclidean'] -= 0.05
            base_weights['dot_product'] -= 0.05
        
        if query_entropy > 3.0:  # 信息熵高，特征分布均匀
            base_weights['weighted_cosine'] += 0.05
            base_weights['cosine'] -= 0.05
        
        # 权重归一化
        total_weight = sum(base_weights.values())
        weights = {k: v/total_weight for k, v in base_weights.items()}
        
        # 高级归一化：使用Sigmoid函数进行平滑归一化
        normalized_sims = {}
        for method, scores in similarities.items():
            if method in weights:
                # 计算分数的统计信息
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                if std_score > 1e-8:
                    # Z-score标准化后使用sigmoid
                    z_scores = (scores - mean_score) / std_score
                    # 使用sigmoid函数将分数映射到(0,1)
                    normalized_scores = 1 / (1 + np.exp(-z_scores))
                else:
                    # 如果标准差太小，使用线性归一化
                    min_score, max_score = np.min(scores), np.max(scores)
                    if max_score > min_score:
                        normalized_scores = (scores - min_score) / (max_score - min_score)
                    else:
                        normalized_scores = np.ones_like(scores) * 0.5
                
                normalized_sims[method] = normalized_scores
        
        # 加权融合
        fused_scores = np.zeros(len(self.commands_data))
        for method, weight in weights.items():
            if method in normalized_sims:
                fused_scores += weight * normalized_sims[method]
        
        return fused_scores
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        执行优化搜索
        
        Args:
            query: 用户查询文本
            top_k: 返回结果数量
            
        Returns:
            匹配的指令列表，按相似度排序
        """
        print(f"\n🔍 正在搜索: '{query}'")
        
        # 生成查询向量
        print("正在生成归一化查询向量...")
        query_vector = self.encode_text(query)
        
        # 验证查询向量归一化
        query_norm = np.linalg.norm(query_vector)
        print(f"查询向量模长: {query_norm:.6f}")
        
        # 计算优化相似度
        print("正在计算优化相似度指标...")
        similarities = self.calculate_similarities_optimized(query_vector)
        
        # 最终权重融合
        print("正在进行最终自适应融合...")
        fused_scores = self.adaptive_weighted_fusion_final(similarities, query_vector)
        
        # 获取top-k结果
        top_indices = np.argsort(fused_scores)[::-1][:top_k]
        
        # 构建结果
        results = []
        for i, idx in enumerate(top_indices):
            result = {
                'rank': i + 1,
                'score': float(fused_scores[idx]),
                'instruction': self.commands_data[idx]['指令'],
                'description': self.commands_data[idx]['描述'],
                'original_text': self.commands_data[idx]['原文'],
                'cosine_similarity': float(similarities['cosine'][idx]),
                'angle_similarity': float(similarities['angle'][idx]),
                'euclidean_similarity': float(similarities['euclidean'][idx]),
                'dot_product': float(similarities['dot_product'][idx]),
                'weighted_cosine': float(similarities['weighted_cosine'][idx]),
                'position_weighted': float(similarities['position_weighted'][idx])
            }
            results.append(result)
        
        return results
    
    def compare_with_original(self, query: str, top_k: int = 5):
        """比较最终版本与原始版本的搜索结果"""
        print(f"\n🔬 搜索算法对比分析 (查询: '{query}')")
        print("="*90)
        
        query_vector = self.encode_text(query)
        similarities = self.calculate_similarities_optimized(query_vector)
        
        # 显示各种单一指标的结果
        methods_info = [
            ('cosine', '余弦相似度', 'cosine'),
            ('angle', '角度相似度', 'angle'),
            ('euclidean', '欧氏距离相似度', 'euclidean'),
            ('dot_product', '点积相似度', 'dot_product'),
            ('weighted_cosine', '加权余弦相似度', 'weighted_cosine'),
            ('position_weighted', '位置加权相似度', 'position_weighted')
        ]
        
        for method_key, method_name, sim_key in methods_info:
            if sim_key in similarities:
                scores = similarities[sim_key]
                top_indices = np.argsort(scores)[::-1][:top_k]
                print(f"\n📊 {method_name} 前{top_k}结果:")
                for i, idx in enumerate(top_indices):
                    instruction = self.commands_data[idx]['指令']
                    score = scores[idx]
                    print(f"  {i+1}. {instruction} (得分: {score:.4f})")
        
        # 最终融合结果
        fused_scores = self.adaptive_weighted_fusion_final(similarities, query_vector)
        top_indices = np.argsort(fused_scores)[::-1][:top_k]
        print(f"\n🎯 最终优化融合前{top_k}结果:")
        for i, idx in enumerate(top_indices):
            instruction = self.commands_data[idx]['指令']
            score = fused_scores[idx]
            print(f"  {i+1}. {instruction} (得分: {score:.4f})")
        
        print("\n" + "="*90)
    
    def interactive_search(self):
        """交互式搜索模式"""
        print("\n" + "="*80)
        print("🎯 SCPI指令智能搜索工具 - 最终优化版本")
        print("="*80)
        print("✨ 专门针对归一化向量优化，确保搜索精确性")
        print("输入您的问题或描述，我将为您找到最匹配的SCPI指令")
        print("")
        print("🎮 可用命令:")
        print("  • 输入问题直接搜索")
        print("  • 'compare <查询>' - 查看详细对比分析")
        print("  • 'help' - 显示帮助信息")
        print("  • 'quit' 或 'exit' - 退出程序")
        print("-"*80)
        
        while True:
            try:
                query = input("\n💬 请输入您的问题: ").strip()
                
                if query.lower() in ['quit', 'exit', '退出', 'q']:
                    print("感谢使用！再见！👋")
                    break
                
                if query.lower() == 'help' or query == '帮助':
                    self.show_help()
                    continue
                
                if query.startswith('compare ') or query.startswith('比较 '):
                    search_query = query.split(' ', 1)[1] if len(query.split(' ', 1)) > 1 else "acquire"
                    self.compare_with_original(search_query)
                    continue
                
                if not query:
                    print("⚠️ 请输入有效的问题！")
                    continue
                
                # 执行搜索
                results = self.search(query, top_k=10)
                
                # 显示结果
                self.display_results(results, query)
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断，正在退出...")
                break
            except Exception as e:
                print(f"\n❌ 搜索过程中出现错误: {e}")
    
    def display_results(self, results: List[Dict], query: str):
        """显示搜索结果"""
        print(f"\n📊 最终优化搜索结果 (查询: '{query}')")
        print("="*100)
        
        for result in results:
            # 使用渐变色emoji表示排名
            rank_emoji = ["🥇", "🥈", "🥉", "🏅", "🎖️"][min(result['rank']-1, 4)]
            
            print(f"\n{rank_emoji} 排名 #{result['rank']} (综合得分: {result['score']:.4f})")
            print(f"📝 指令: {result['instruction']}")
            print(f"📖 描述: {result['description']}")
            
            # 智能截断原文显示
            original = result['original_text']
            if len(original) > 150:
                # 尝试在句号处截断
                truncated = original[:150]
                last_period = truncated.rfind('.')
                if last_period > 100:
                    truncated = original[:last_period+1]
                else:
                    truncated = original[:150] + "..."
            else:
                truncated = original
            
            print(f"📄 原文: {truncated}")
            print(f"📈 详细相似度指标:")
            print(f"   • 余弦相似度: {result['cosine_similarity']:.4f}")
            print(f"   • 角度相似度: {result['angle_similarity']:.4f}")
            print(f"   • 欧氏相似度: {result['euclidean_similarity']:.4f}")
            print(f"   • 点积相似度: {result['dot_product']:.4f}")
            print(f"   • 加权余弦: {result['weighted_cosine']:.4f}")
            print(f"   • 位置加权: {result['position_weighted']:.4f}")
            print("-" * 100)
    
    def show_help(self):
        """显示帮助信息"""
        print("\n" + "="*80)
        print("📖 最终优化版SCPI搜索工具帮助")
        print("="*80)
        print("🎯 这是专门针对归一化向量优化的SCPI指令搜索工具")
        print("")
        print("🔍 搜索示例:")
        print("  • '如何设置采集模式'")
        print("  • '查看频率测量'")
        print("  • '触发配置'")
        print("  • 'acquire measurement'")
        print("  • 'trigger settings'")
        print("")
        print("⚙️ 最终优化特性:")
        print("  🧠 智能向量匹配:")
        print("    • 自动归一化查询向量")
        print("    • 针对单位向量优化的距离计算")
        print("    • 考虑向量分量重要性差异")
        print("")
        print("  📊 6种相似度计算:")
        print("    • 余弦相似度 - 语义相似性")
        print("    • 角度相似度 - 向量夹角")
        print("    • 欧氏距离相似度 - 几何距离(归一化优化)")
        print("    • 点积相似度 - 向量投影")
        print("    • 加权余弦相似度 - 特征重要性加权")
        print("    • 位置加权相似度 - 考虑特征位置")
        print("")
        print("  🎛️ 自适应权重:")
        print("    • 根据查询稀疏性调整")
        print("    • 考虑主要特征强度")
        print("    • 基于信息熵动态优化")
        print("")
        print("🎮 特殊命令:")
        print("  • 'compare <查询>' - 详细对比分析")
        print("  • 'help' - 显示此帮助")
        print("  • 'quit' - 退出程序")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='SCPI指令智能搜索工具 - 最终优化版本')
    parser.add_argument('--json_file', '-f', default='all_scpi_commands.json',
                       help='SCPI指令JSON文件路径')
    parser.add_argument('--model', '-m', default='sentence-transformers/all-MiniLM-L6-v2',
                       help='使用的transformer模型')
    parser.add_argument('--query', '-q', help='直接执行查询（非交互模式）')
    parser.add_argument('--top_k', '-k', type=int, default=10,
                       help='返回结果数量')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='显示详细对比分析')
    
    args = parser.parse_args()
    
    try:
        # 初始化搜索器
        searcher = SCPICommandSearcherFinal(args.json_file, args.model)
        
        if args.query:
            if args.compare:
                # 对比分析模式
                searcher.compare_with_original(args.query, args.top_k)
            else:
                # 直接查询模式
                results = searcher.search(args.query, args.top_k)
                searcher.display_results(results, args.query)
        else:
            # 交互模式
            searcher.interactive_search()
            
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 