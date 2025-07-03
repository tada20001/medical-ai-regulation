# main.py 또는 app.py 파일의 최종 완성본 (UI 순서 수정)

import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from backend.core import run_llm
from typing import List, Dict, Any, Set

load_dotenv()

# --- 1. 페이지 기본 설정 및 사이드바 ---
st.set_page_config(page_title="의료 AI 규제 검색 시스템", layout="wide")

with st.sidebar:
    st.header("API Key 설정")
    openai_api_key_input = st.text_input(
        "OpenAI API Key", type="password", key="openai_key_input",
        placeholder="sk-xxxxxxxx"
    )
    st.info("⚠️ OpenAI API 키를 입력해야 챗봇을 사용할 수 있습니다.")

# --- 2. 메인 타이틀 ---
st.title("의료 AI 규제 검색 시스템")
st.info("MDCG 가이던스 문서 내용을 기반으로 질문에 답변합니다.")

# --- 3. 세션 상태 초기화 ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 4. 참고 자료 포맷팅 함수 ---
def create_source_string(source_docs: List[Dict[str, Any]]) -> str:
    if not source_docs: return ""
    sources_set = set()
    for doc in source_docs:
        title = doc.metadata.get('title', 'N/A').replace('*', '')
        ref = doc.metadata.get('reference', 'N/A')
        page = int(doc.metadata.get('page', 0)) + 1
        url = doc.metadata.get('original_source', '#')
        sources_set.add(f'<li><a href="{url}" target="_blank">{title}</a> (Reference: {ref}, Page: {page})</li>')
    sources_list = sorted(list(sources_set))
    return "<b>참고 자료:</b><ul>" + "".join(sources_list) + "</ul>"

# --- 5. 질문 처리 로직 (버튼 클릭 시 실행될 내용) ---
# [핵심] 폼이 제출되었는지 먼저 확인하고 로직을 실행합니다.
if 'form_submitted' in st.session_state and st.session_state.form_submitted:
    prompt = st.session_state.prompt_input

    # API 키 유효성 검사
    if not st.session_state.openai_api_key:
        st.error("사이드바에 OpenAI API 키를 먼저 입력해주세요!")
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.spinner("답변을 생성하는 중입니다..."):
            history_for_llm = [msg for msg in st.session_state.chat_history if msg['role'] != 'assistant'][:-1]
            generated_response = run_llm(query=prompt, chat_history=history_for_llm)
            sources_str = create_source_string(generated_response['source_documents'])
            formatted_response = f"{generated_response['result']}<br><br>{sources_str}"
            st.session_state.chat_history.append({"role": "assistant", "content": formatted_response})

    # 처리 후 상태 초기화
    st.session_state.form_submitted = False
    st.session_state.prompt_input = ""
    st.rerun()


# --- 6. 이전 대화 내용 표시 ---
# 대화 기록을 먼저 화면에 그립니다.
for i, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")
    else:
        message(msg["content"], is_user=False, key=f"assistant_{i}", allow_html=True)


# --- 7. 질문 입력창 (맨 아래 위치) ---
# [핵심] 입력창을 맨 마지막에 배치합니다.
with st.form(key="chat_form"):
    st.text_area(
        "질문 입력:",
        key="prompt_input",
        height=150,
        placeholder="생성형 AI 규제에 대해 설명해보세요..."
    )
    # form_submit_button은 form 내부 상태를 변경하는 역할을 합니다.
    if st.form_submit_button(label="전송"):
        st.session_state.form_submitted = True
        st.session_state.openai_api_key = openai_api_key_input # 전송 시점의 키를 저장
        st.rerun()
