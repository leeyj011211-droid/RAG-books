import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="프리미엄 AI 도서관", page_icon="📚")
st.title("📚 지능은 높고 속도는 빠른 AI 사서")
st.caption("추천 받고자 하는 주제를 설명하면 3가지의 책을 추천해줍니다.")

# 1. 리소스 로드 (캐싱 적용)
@st.cache_resource
def load_resources():
    model_name = "intfloat/multilingual-e5-small"
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
    
    vector_db = FAISS.load_local("faiss_book_index", embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatGroq(
        api_key="Key!!!!!!!!", # 제공해주신 키 사용 (보안상 앞부분만 표시)
        model_name="llama-3.1-8b-instant",
        temperature=0.5
    )
    
    return vector_db.as_retriever(search_kwargs={"k": 3}), llm

@st.cache_data
def load_origin_df():
    df = pd.read_csv('./dataset/google_books_dataset.csv')
    df['thumbnail'] = df['thumbnail'].fillna('') # NaN 미리 처리
    return df

retriever, llm = load_resources()
df_origin = load_origin_df()

# app.py 상단에 추가
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1903/1903162.png", width=100) # 도서관 로고 느낌
    st.title("Library Settings")
    st.info("현재 15,000권의 장서가 등록되어 있습니다.")
    
    # 온도 조절 슬라이더 (사용자가 직접 정밀도 조절)
    temp = st.slider("사서의 창의성 (Temperature)", 0.0, 1.0, 0.0, 0.1)
    
    st.divider()
    st.markdown("### 💡 검색 팁")
    st.caption("- 특정 장르를 말씀해 보세요.\n- 기분에 맞는 책을 물어보세요.")
    
# --- 이미지 검색 함수 ---
def get_book_thumbnail(title):
    try:
        # 공백 제거 후 비교하여 매칭 확률 업
        target_row = df_origin[df_origin['title'].str.strip() == str(title).strip()]
        if not target_row.empty:
            url = target_row['thumbnail'].values[0]
            if isinstance(url, str) and url.startswith('http'):
                return url
        return None
    except:
        return None

# 2. RAG 체인 설정
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 오직 제공된 [도서 정보] 리스트에 있는 책들로만 답변하는 전문 사서야.
    사용자의 입력에 따라 아래와 같이 다르게 행동해줘.

    [행동 규칙]
    1. **단순 인사나 일상 대화**: 사용자가 "안녕", "반가워", "누구니?" 등의 인사를 하면 책을 추천하지 말고, 친절하고 짧게 인사를 건네며 무엇을 도와줄지 물어봐.
    2. **책 추천 요청**: 책에 대한 질문이나 추천 요청이 있을 때만 반드시 **3권의 책**을 선정하여 아래 [출력 형식]으로 답변해줘.
    3. 만약 제공된 정보 중에 사용자의 질문과 관련된 책이 하나도 없다면, "죄송합니다. 현재 저희 도서관 데이터에는 관련 도서가 없습니다."라고 답변해.
    4. 추천하는 책의 제목은 반드시 데이터셋에 적힌 그대로 원문으로 표기해.
    5. 추천 도서의 제목은 영문 제목이 있으면 영문을 우선하고, 없으면 한글을 써.

    [추천 도서 출력 형식]
    1. **도서 제목** (저자) 
    - 🏷️ **핵심 키워드**: #키워드1 #키워드2
    - 📝 **한 줄 요약**: 핵심 내용과 추천 이유 정리
    (항목 간 줄바꿈을 철저히 지켜서 가독성을 높여줘.)

    마지막에 "더 궁금한 책이 있으신가요?"라고 짧게 마무리해줘."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "도서 정보: {context}\n\n사용자 질문: {input}")
])
rag_chain = (
    RunnablePassthrough.assign(context=lambda x: "\n\n".join([d.page_content for d in retriever.invoke(x["input"])]))
    | qa_prompt | llm | StrOutputParser()
)

        
# 3. 채팅 UI 및 로직
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 내역 출력
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 사용자 입력 처리
if prompt := st.chat_input("책에 대해 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. 문서 먼저 검색 (이게 반드시 위에 있어야 docs 정의 에러가 안 남)
        docs = retriever.invoke(prompt) 
        
        response_placeholder = st.empty()
        full_response = ""
        
        # 2. 답변 생성
        for chunk in rag_chain.stream({"input": prompt, "chat_history": st.session_state.messages[:-1]}):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
        
        # 3. 이미지 출력 (인사말이 아닐 때만)
        if "1." in full_response:
            st.write("---")
            st.markdown("#### 📖 추천 도서 이미지")
            
            # 답변 텍스트에서 [제목] 형태를 추출하거나, 
            # 검색된 docs 중 답변에 이름이 언급된 책들만 필터링
            recommended_docs = [d for d in docs if d.metadata.get('title') in full_response]
            
            if recommended_docs:
                cols = st.columns(len(recommended_docs))
                for i, doc in enumerate(recommended_docs):
                    title = doc.metadata.get('title')
                    img_url = get_book_thumbnail(title)
                    with cols[i]:
                        if img_url:
                            st.image(img_url, use_container_width=True)
                        else:
                            st.info("이미지 준비 중")
                        st.caption(f"**{title}**")

    st.session_state.messages.append({"role": "assistant", "content": full_response})