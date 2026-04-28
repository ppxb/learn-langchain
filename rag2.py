from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("./data/rag2.txt", encoding="utf-8")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)

chunks = text_splitter.split_documents(docs)
print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前5个块内容示例 ---")

for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')
