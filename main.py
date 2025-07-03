# main.py 또는 app.py 파일의 최종 완성본

import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from backend.core import run_llm
from typing import List, Dict, Any, Set # 타입 힌트 import

load_dotenv()

# --- 1. 페이지 기본 설정 ---
st.set_page_config(page_title="의료 AI 규제 검색 시스템", layout="centered")
st.title("의료 AI 규제 검색 시스템")
st.info("MDCG 가이던스 문서 내용을 기반으로 질문에 답변합니다. 첫 질문은 다소 시간이 걸릴 수 있습니다.")


# --- 2. 채팅 기록을 위한 세션 상태 초기화 ---
# [개선] 이제 'chat_history' 하나로만 모든 대화를 관리합니다.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- 3. UI 렌더링: 이전 대화 내용 표시 ---
# [핵심] 입력창보다 먼저, 저장된 모든 메시지를 화면에 그립니다.
for i, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")
    else:
        # AI의 답변에는 참고 자료까지 포함되어 있습니다.
        message(msg["content"], is_user=False, key=f"assistant_{i}", allow_html=True)


# --- 4. 함수 정의: 참고 자료 포맷팅 ---
def create_source_string(source_docs: List[Dict[str, Any]]) -> str:
    if not source_docs:
        return ""

    sources_set = set()
    for doc in source_docs:
        title = doc.metadata.get('title', 'N/A').replace('*', '') # Bold 마크다운 제거
        ref = doc.metadata.get('reference', 'N/A')
        page = int(doc.metadata.get('page', 0)) + 1
        url = doc.metadata.get('original_source', '#')

        # 보기 좋은 HTML 링크 형태로 조합
        sources_set.add(f'<li><a href="{url}" target="_blank">{title}</a> (Reference: {ref}, Page: {page})</li>')

    if not sources_set:
        return ""

    sources_list = sorted(list(sources_set))
    return "<b>참고 자료:</b><ul>" + "".join(sources_list) + "</ul>"


# --- 5. UI 렌더링: 질문 입력창 ---
# [핵심] st.chat_input을 사용하여 화면 하단에 고정되는 입력창을 만듭니다.
with st.form(key="chat_form", clear_on_submit=True):
    prompt = st.text_area("질문 입력:", key="prompt_input", height=150, placeholder="생성형 AI 규제에 대해 설명해보세요...")

    # 사용자의 질문을 기록하고 즉시 화면에 표시
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)

    # AI 응답 생성
    with st.spinner("답변을 생성하는 중입니다..."):
        # run_llm에 전달할 채팅 기록 포맷팅 (마지막 질문 제외)
        history_for_llm = [msg for msg in st.session_state.chat_history if msg['role'] != 'assistant'][:-1]

        # 백엔드 엔진 호출
        generated_response = run_llm(query=prompt, chat_history=history_for_llm)

        # 참고 자료 문자열 생성 (HTML 링크 포함)
        sources_str = create_source_string(generated_response['source_documents'])

        # 최종 답변 포맷팅
        formatted_response = f"{generated_response['result']}<br><br>{sources_str}"

        # AI의 답변을 기록하고 화면에 표시
        st.session_state.chat_history.append({"role": "assistant", "content": formatted_response})
        st.rerun() # 화면을 새로고침하여 AI 답변을 즉시 그리도록 함
