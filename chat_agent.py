from dotenv import load_dotenv
import google.generativeai as genai
import os
import jieba
import re
import logging
from typing import List, Dict, Any
from langchain.memory import ConversationBufferMemory
from utils.pdf_loader import load_pdfs
from langchain_huggingface import HuggingFaceEmbeddings
from utils.text_splitter.literary_splitter import LiteraryTextSplitter
import torch

load_dotenv()
jieba.setLogLevel(logging.WARNING)

class ChatAgent:
    def __init__(self, pdf_dir):
        # Gemini API 配置
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key, transport="rest")

        # 生成参数配置
        self.model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-thinking-exp-01-21',
            generation_config={
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 40,
                "max_output_tokens": 65536,
            }
        )
        # self.model = genai.GenerativeModel(
        #     model_name='gemini-2.0-pro-exp',
        #     generation_config={
        #         "temperature": 0.2,
        #         "top_p": 1,
        #         "top_k": 40,
        #         "max_output_tokens": 8192,
        #     }
        # )

        # 初始化知识库和对话历史, 使用 LiteraryTextSplitter
        self.vectorstore = load_pdfs(pdf_dir, splitter=LiteraryTextSplitter())
        self.chat_history = []
        self.max_history = 10

        # 初始化embedding模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="TencentBAC/Conan-embedding-v1",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder="./embeddings_cache"
        )

    def chat(self, query):
        try:
            # 使用Milvus进行搜索
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 16}
            }
            # 先将query转换为embedding向量
            query_embedding = self.embeddings.embed_query(query)
            results = self.vectorstore.search(
                [query_embedding], "embedding", search_params, limit=12, output_fields=["source", "chunk_index", "chunk_total", "chunk_size", "chunk_overlap", "content"]
            )

            # Milvus返回的结果处理
            filtered_docs = []
            for hits in results:  # results 的第一个元素是 Hits 对象
                for hit in hits:
                    doc = {
                        'page_content': hit.get('content'),
                        'metadata': {
                            'source': hit.get('source'),
                            'chunk_index': hit.get('chunk_index'),
                            'chunk_total': hit.get('chunk_total'),
                            'chunk_size': hit.get('chunk_size'),
                            'chunk_overlap': hit.get('chunk_overlap')
                        }
                    }
                    filtered_docs.append(doc)

            
            # 动态上下文管理
            context = ""
            current_length = 0
            max_length = min(8000, 3000 + len(query) * 10)
            
            for doc in filtered_docs:
                content = doc['page_content']
                if current_length + len(content) > max_length:
                    break
                context += f"\n\n{content}"
                current_length += len(content)
            
            # 构建提示词
            history_text = ""
            if self.chat_history:
                history_text = "\n历史对话：\n" + "\n".join(
                    f"问：{q}\n答：{a}" for q, a in self.chat_history
                )
            prompt = f"""你是一位文学作品解读助手，你的回答应当帮助学习者深入理解作品。不要有任何开场白或过渡语，只输出正文。

你是一位资深的文学评论家，有自己独到的见解，擅长解读各种文学作品，包括小说、诗歌、散文、戏剧等。你的任务是帮助读者深入理解作品的内涵、艺术价值和社会意义。

在回答问题时，请注意以下几个方面：

1.  **主题分析：** 准确概括作品的主题，并分析主题在作品中的表现方式。例如，可以通过情节、人物、意象等元素来揭示主题。

2.  **人物分析：** 分析主要人物的性格、动机、命运，以及人物之间的关系。注意人物的复杂性和多面性，避免简单化和标签化。

3.  **情节分析：** 梳理作品的情节发展，分析情节的起承转合、高潮和结局。注意情节的逻辑性和合理性，以及情节对主题的贡献。

4.  **意象分析：** 识别作品中重要的意象，并分析其象征意义。注意意象的文化内涵和情感色彩，以及意象对主题的烘托。

5.  **语言分析：** 分析作品的语言风格，例如修辞手法、句式特点、用词选择等。注意语言的艺术性和表现力，以及语言对人物塑造和情感表达的作用。

6.  **背景分析：** 分析作品的创作背景，包括社会背景、历史背景、文化背景等。注意背景对作品主题和人物的影响。

7.  **流派分析：** 确定作品所属的文学流派，并分析其流派特点。注意流派对作品风格和主题的影响。

8.  **社会意义：** 探讨作品的社会意义，例如对现实的反映、对人性的思考、对未来的展望等。注意作品的批判性和启示性。

在回答问题时，请结合以上几个方面，进行全面而深入的分析。避免泛泛而谈，要结合具体的文本细节进行论证。可以使用专业的文学术语，但要确保读者能够理解。
不要刻意地结构化输出，要保持回答的自然流畅，要有自己的文风，解读作品何尝不是一种写作。
请记住，你的目标是帮助读者更好地理解作品，而不是炫耀你的知识。因此，要保持谦逊和客观的态度，尊重不同的观点和解读。
终端无法渲染markdown，所以在输出时，请不要使用 Markdown 语法。请使用以下方式组织你的回答：
- 使用 '-' 作为主要分隔符。
- 使用数字列表来列举要点。
- 使用 '#' 作为标题。
- 不要用使用*

基于以下参考资料和历史对话回答问题:
参考资料:
{context}

历史对话:
{history_text}

问题：{query}"""

            # 生成响应
            response = self.model.generate_content(prompt)
            self.chat_history.append((query, response.text))
            if len(self.chat_history) > self.max_history:
                self.chat_history = self.chat_history[-self.max_history:]
            
            yield response.text
            
        except Exception as e:
            yield f"错误: {str(e)}"
    
    def _evaluate_doc_quality(self, content: str, query: str) -> float:
        """评估文档质量"""
        indicators = {
            'term_count': len(re.findall(r'[A-Z][a-z]+(?:[ -][A-Z][a-z]+)*', content)),
            'structure_score': 1 if re.search(r'。|；|！|？', content) else 0.5,
            'relevance_score': sum(q in content for q in jieba.cut(query)) / len(query)
        }
        
        weights = {'term_count': 0.3, 'structure_score': 0.3, 'relevance_score': 0.4}
        final_score = sum(score * weights[metric] for metric, score in indicators.items())
        
        return min(1.0, final_score)
