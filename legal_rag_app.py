import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import requests
import os

KNOWLEDGE_PATH = r'D:\周儒轩\周儒轩\大学文件\大四\2025大四上\项目-法律大模型\知识库'
DESKTOP = os.path.expanduser('~/Desktop')
JSON_FILES = [f for f in os.listdir(DESKTOP) if f.endswith('.json')]
JSON_PATH = os.path.join(DESKTOP, JSON_FILES[0])

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read()

st.set_page_config(page_title='行政投诉信专家', layout='centered')
st.title('🧑‍⚖️ 行政投诉信撰写专家（极速版：qwen2.5:7b-instruct）')

with st.sidebar:
    st.header('配置')
    knowledge_path = st.text_input('知识库路径', KNOWLEDGE_PATH)
    if st.button('重建/更新向量索引'):
        with st.spinner('构建索引（首次1-3分钟）...'):
            loader = DirectoryLoader(knowledge_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name='DMetaSoul/Dmeta-embedding-zh')
            vectordb = Chroma.from_documents(splits, embeddings, persist_directory='./chroma_db')
            vectordb.persist()
            st.success('索引构建完成！')

st.write('请输入案情描述：')
user_input = st.text_area('', height=150, placeholder='例如：小区电梯故障一个月，物业不修也不公开维保记录')

if st.button('生成投诉信'):
    if user_input:
        with st.spinner('检索+生成（30-60秒）...'):
            embeddings = HuggingFaceEmbeddings(model_name='DMetaSoul/Dmeta-embedding-zh')
            vectordb = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)
            retriever = vectordb.as_retriever(search_kwargs={'k': 5})
            context_docs = retriever.invoke(user_input)
            context = '\n\n'.join([doc.page_content for doc in context_docs])

            payload = {
                'model': 'qwen2.5:7b-instruct',
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': f'相关法律资料：\n{context}\n\n用户问题：{user_input}'}
                ],
                'stream': False,
                'temperature': 0.1
            }
            r = requests.post('http://localhost:11434/api/chat', json=payload, timeout=300)
            if r.status_code == 200:
                st.success('生成成功！')
                st.markdown(r.json()['message']['content'])
            else:
                st.error(f'模型错误：{r.text}')
    else:
        st.warning('请先输入案情')

st.caption('Powered by qwen2.5:7b-instruct + Chroma + Streamlit')
