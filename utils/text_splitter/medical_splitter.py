from dataclasses import dataclass
import re
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class TextStats:
    avg_sentence_length: float
    avg_paragraph_length: float
    term_density: float

class AdaptiveMedicalSplitter:
    def __init__(self):
        self.term_patterns = [
            r'[A-Z][a-z]+(?:[ -][A-Z][a-z]+)*',  # 医学术语
            r'\d+(?:\.\d+)?%?(?:[ -][A-Za-z]+)+', # 剂量/数值
            r'[A-Z]{2,}',  # 缩写
        ]
        
        self.section_patterns = {
            'chapter': r'^第[一二三四五六七八九十百]+章|\d+\s*章',
            'section': r'^\d+\.\d+\s+[^\n]+',
            'subsection': r'^\d+\.\d+\.\d+\s+[^\n]+'
        }

    def analyze_text(self, text: str) -> TextStats:
        """分析文本特征"""
        sentences = re.split(r'[。！？]', text)
        paragraphs = text.split('\n\n')
        
        avg_sent_len = sum(len(s) for s in sentences) / max(len(sentences), 1)
        avg_para_len = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        
        term_count = sum(len(re.findall(pattern, text)) for pattern in self.term_patterns)
        term_density = term_count / len(text) if text else 0
        
        return TextStats(avg_sent_len, avg_para_len, term_density)

    def get_optimal_chunk_size(self, stats: TextStats) -> Dict[str, int]:
        """根据文本统计信息获取最佳块大小"""
        if stats.term_density > 0.05:
            chunk_size = 1318
            chunk_overlap = 330
        else:
            chunk_size = 871
            chunk_overlap = 218
        
        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }

    def split_document(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        stats = self.analyze_text(text)
        params = self.get_optimal_chunk_size(stats)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            is_separator_regex=False
        )
        
        def clean_chunk(chunk: str) -> str:
            if not chunk.endswith(("。", "！", "？")):
                last_period = max(chunk.rfind("。"), chunk.rfind("！"), chunk.rfind("？"))
                if last_period != -1:
                    chunk = chunk[:last_period + 1]
            return chunk.strip()
        
        chunks = splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            clean_text = clean_chunk(chunk)
            if not clean_text:
                continue
                
            # 修改元数据格式，只保留简单类型
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": str(i),  # 转换为字符串
                "chunk_total": str(len(chunks)),  # 转换为字符串
                "chunk_size": str(params["chunk_size"]),  # 转换为字符串
                "chunk_overlap": str(params["chunk_overlap"])  # 转换为字符串
            }
            
            documents.append(Document(
                page_content=clean_text,
                metadata=chunk_metadata
            ))
            
        return documents