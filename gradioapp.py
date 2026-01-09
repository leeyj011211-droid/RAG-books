import gradio as gr
import pandas as pd
import re
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# --- [1] 리소스 로드 및 전처리 ---
MY_GROQ_KEY = "Key!!!!!!!!" # 본인의 실제 키 입력

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small", encode_kwargs={'normalize_embeddings': True})
vector_db = FAISS.load_local("faiss_book_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(api_key=MY_GROQ_KEY, model_name="llama-3.1-8b-instant", temperature=0.0)

# 데이터 로드 및 결측치 방어
df_origin = pd.read_csv('./dataset/google_books_dataset.csv')
df_origin['thumbnail'] = df_origin['thumbnail'].fillna('')
# 제목(title) 열에 숫자가 있거나 비어있는 경우를 대비해 모두 문자열로 강제 변환
df_origin['title'] = df_origin['title'].astype(str).fillna('Unknown Title')

def get_book_info(title_to_find):
    if not title_to_find or pd.isna(title_to_find):
        return None
    
    # [에러 방지 핵심] 입력된 제목을 문자열로 변환하고 양끝 공백 제거
    search_title = str(title_to_find).strip()
    
    # 데이터셋에서 매칭 (대소문자 무시하지 않고 정확히 일치 확인)
    target = df_origin[df_origin['title'].str.strip() == search_title]
    
    if not target.empty:
        url = target['thumbnail'].values[0]
        if isinstance(url, str) and url.startswith('http'):
            return url
    return None

# --- [2] 프롬프트 수정 (Streamlit 스타일 적용) ---
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 15,000권의 장서를 보유한 도서관의 '수석 사서'야. 
    [출력 규칙]
    1. 반드시 제공된 도서 정보 내에서만 답변해.
    2. 책 제목은 무조건 **[한글 제목 (English Title)]** 형식으로 표기해.
    3. 추천은 항상 3권을 추천 하고, 각 책마다 추천 이유를 사서처럼 친절하게 설명해줘.
    4. 답변은 한국어로 하되, 전문적이고 따뜻한 느낌을 유지해."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "도서 정보: {context}\n\n사용자 질문: {input}")
])
rag_chain = qa_prompt | llm | StrOutputParser()

# --- [3] 응답 함수 ---
def respond(message, chat_history):
    docs = retriever.invoke(message)
    context = "\n\n".join([d.page_content for d in docs])
    
    history_langchain = []
    for msg in chat_history:
        role = "human" if msg["role"] == "user" else "ai"
        history_langchain.append((role, msg["content"]))
    
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})
    
    full_response = ""
    for chunk in rag_chain.stream({"input": message, "chat_history": history_langchain, "context": context}):
        full_response += chunk
        chat_history[-1]["content"] = full_response
        yield chat_history, [] 

    # 이미지 매칭 (에러 방어형)
    images = []
    # 텍스트 내에서 [제목 (영어)] 패턴 추출
    pattern = r'\[(.*?)\s*\((.*?)\)\]'
    found_titles = re.findall(pattern, full_response)
    
    for kor_t, eng_t in found_titles:
        img_url = get_book_info(kor_t)
        if img_url:
            images.append((img_url, kor_t))
    
    # 혹시 패턴으로 못 찾았다면 Docs 자체 제목으로 재시도
    if not images:
        for d in docs:
            t = d.metadata.get('title')
            url = get_book_info(t)
            if url: images.append((url, t))
            
    yield chat_history, images[:3]

# --- [4] 고급 UI 디자인 (CSS) ---
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Noto Sans KR"), "ui-sans-serif", "system-ui"],
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("<h1 style='text-align: center; color: #1a365d;'>📚 AI 지능형 서가</h1>")
    gr.Markdown("<p style='text-align: center;'>당신의 취향을 분석하여 최적의 도서를 큐레이팅합니다.</p>")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="사서와 대화", height=600)
            with gr.Row():
                msg = gr.Textbox(placeholder="어떤 책을 원하시나요? (예: 위로가 되는 소설)", show_label=False, scale=9)
                submit_btn = gr.Button("보내기", variant="primary", scale=1)
            clear = gr.Button("대화 초기화", variant="secondary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📖 추천 도서 갤러리")
            gallery = gr.Gallery(label="표지", columns=1, rows=3, height=600, object_fit="contain")

    # 이벤트 연결
    msg.submit(respond, [msg, chatbot], [chatbot, gallery])
    msg.submit(lambda: "", None, msg)
    submit_btn.click(respond, [msg, chatbot], [chatbot, gallery])
    submit_btn.click(lambda: "", None, msg)
    clear.click(lambda: ([], []), None, [chatbot, gallery])

if __name__ == "__main__":
    demo.launch()