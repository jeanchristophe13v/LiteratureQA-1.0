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

class LiteraryTextSplitter:
    def __init__(self):
        self.sentence_separators = r'[。！？\n]'  # 句子的分隔符
        self.paragraph_separators = r'\n\n+'  # 段落的分隔符
        self.chapter_separators = r'(第[一二三四五六七八九十百千]+章|第\s*\d+\s*节|episode\s*\d+)'  # 章节的分隔符

    def analyze_text(self, text: str) -> TextStats:
        """分析文本特征"""
        sentences = re.split(self.sentence_separators, text)
        paragraphs = re.split(self.paragraph_separators, text)

        avg_sent_len = sum(len(s) for s in sentences) / max(len(sentences), 1)
        avg_para_len = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)

        #  文学作品术语密度较低，可以忽略
        term_density = 0

        return TextStats(avg_sent_len, avg_para_len, term_density)

    def get_optimal_chunk_size(self, stats: TextStats) -> Dict[str, int]:
        """根据文本统计信息获取最佳块大小"""
        #  文学作品通常结构清晰，可以采用较大的块大小
        chunk_size = 2048
        chunk_overlap = 256

        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }

    def split_document(self, text: str, metadata: Dict[str, Any] = None, max_length: int = 5000) -> List[Document]:
        stats = self.analyze_text(text)
        params = self.get_optimal_chunk_size(stats)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            length_function=len,
            separators=[self.paragraph_separators, self.sentence_separators, " ", ""],
            is_separator_regex=True  # 启用正则表达式分隔符
        )

        chunks = splitter.split_text(text)
        documents = []

        for i, chunk in enumerate(chunks):
            clean_text = chunk.strip()
            if not clean_text:
                continue
            
            # 限制文本块的最大长度
            clean_text = clean_text[:max_length]

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