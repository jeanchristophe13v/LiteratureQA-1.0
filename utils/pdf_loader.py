from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
from typing import List
from tqdm import tqdm
from utils.text_splitter.medical_splitter import AdaptiveMedicalSplitter

def get_pdf_files(pdf_dir):
    """获取目录下的所有PDF文件"""
    return {f for f in os.listdir(pdf_dir) if f.endswith('.pdf')}

def print_knowledge_base_info(current_files, new_files=None):
    """打印知识库信息"""
    print("\n" + "="*50)
    print("知识库包含以下文件：")
    for file in sorted(current_files):  # 按字母顺序排
        status = "[新文件]" if new_files and file in new_files else "[已加载]"
        print(f"  {status} {file}")
    print("="*50 + "\n")

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
from typing import List, Optional
from tqdm import tqdm
from utils.text_splitter.medical_splitter import AdaptiveMedicalSplitter
from utils.text_splitter.literary_splitter import LiteraryTextSplitter
import torch
import time


def get_pdf_files(pdf_dir):
    """获取目录下的所有PDF文件"""
    return {f for f in os.listdir(pdf_dir) if f.endswith('.pdf')}

def print_knowledge_base_info(current_files, new_files=None):
    """打印知识库信息"""
    print("\n" + "="*50)
    print("知识库包含以下文件：")
    for file in sorted(current_files):  # 按字母顺序排
        status = "[新文件]" if new_files and file in new_files else "[已加载]"
        print(f"  {status} {file}")
    print("="*50 + "\n")

def load_pdfs(pdf_dir: str, splitter: Optional[object] = None):
    print("开始加载embedding模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="TencentBAC/Conan-embedding-v1",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder="./embeddings_cache"
    )
    print("embedding模型加载完成")

    current_files = get_pdf_files(pdf_dir)
    if not current_files:
        print(f"\n警告: {pdf_dir} 目录下没有找到PDF文件")
        return None

    connections.connect("default", host="localhost", port="19530")
    collection_name = "literary_knowledge_base"

    # 获取已处理的文件名
    processed_files = set()
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        # Milvus 2.4+ 需要先 load 才能 query
        collection.load()
        # 假设source字段存储了文件名
        for entity in collection.query(expr='id >= 0', output_fields=['source']):
            processed_files.add(entity['source'])

    if set(current_files) != processed_files:
        if utility.has_collection(collection_name):
            print("检测到PDF文件变化，删除旧的Milvus集合...")
            utility.drop_collection(collection_name)

        print("\n创建新的Milvus向量库...")
        print("定义Milvus集合模式...")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="chunk_index", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="chunk_total", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="chunk_size", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="chunk_overlap", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1792)
        ]
        schema = CollectionSchema(fields, "Literary knowledge base")  # 更新集合描述
        collection = Collection(collection_name, schema)

        print_knowledge_base_info(current_files)
        
        # 如果未提供 splitter，则使用默认的 LiteraryTextSplitter
        if splitter is None:
            splitter = LiteraryTextSplitter()
            
        documents = []

        print("\n开始处理文档...")
        for file in tqdm(current_files, desc="加载PDF"):
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
            raw_docs = loader.load()

            full_text = "\n\n".join(doc.page_content for doc in raw_docs)
            processed_docs = splitter.split_document(
                full_text,
                metadata={"source": file},
            )
            documents.extend(processed_docs)

        print(f"\n总计分割为 {len(documents)} 个文本块")

        print("将数据插入Milvus...")
        data = [
            [doc.metadata['source'] for doc in documents],
            [doc.metadata['chunk_index'] for doc in documents],
            [doc.metadata['chunk_total'] for doc in documents],
            [doc.metadata['chunk_size'] for doc in documents],
            [doc.metadata['chunk_overlap'] for doc in documents],
            [doc.page_content for doc in documents],
            embeddings.embed_documents([doc.page_content for doc in documents])
        ]

        collection.insert(data)
        print("创建索引...")
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)
        while not collection.has_index():
            print("等待索引创建完成...")
            time.sleep(1)
        collection.load()
        print("Milvus向量化完成")
        return collection
    else:
        print("未检测到PDF文件变化，使用现有Milvus向量库...")
        return Collection(collection_name)