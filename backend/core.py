from dotenv import load_dotenv
load_dotenv()
from typing import Dict, Any, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

INDEX_NAME = "medical-ai-regulation"

korean_rephrase_prompt = ChatPromptTemplate.from_template(
"""주어진 채팅 기록과 마지막 사용자 질문을 바탕으로, 채팅 기록 없이도 이해할 수 있는 독립적인 질문을 만드세요. 
당신의 유일한 임무는 '사용자 질문'의 핵심 의도를 파악하여, 그것을 독립적인 질문으로 바꾸는 것입니다.

채팅 기록은 오직 '그', '저것'과 같은 대명사가 무엇을 가리키는지 파악하는 데에만 사용하세요.
채팅 기록에 나온 새로운 주제에 대해 질문을 만들지 마세요.
절대 원래 질문에 답하지 마세요.

채팅 기록:
{chat_history}

사용자 질문:
{input}

독립적인 질문:""")


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Dict[str, Any]:
    # --- 1. 기본 설정 (LLM, Embeddings, DB) ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    # as_retriever를 호출할 때, 검색할 네임스페이스를 지정합니다.
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 5, 'namespace': 'mdcg'}  # k: 5개의 관련 문서를 찾음
    )

    # --- 2. 질문 재작성(Rephrasing)을 위한 체인 구성 ---
    #rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    rephrase_prompt = korean_rephrase_prompt
    history_aware_retriever_chain = create_history_aware_retriever(
        llm=chat,
        retriever=retriever, # 위에서 새로 만든 retriever를 사용
        prompt=rephrase_prompt,
        # 'chat_history'는 여기서 전달하지 않습니다.
    )

    # --- 3. 질문 재작성으로 검색된 문서 기반 답변 생성을 위한 체인 구성 ---
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm=chat, prompt=retrieval_qa_chat_prompt)

    # --- 4. 두 체인을 최종적으로 결합 ---
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever_chain, combine_docs_chain=stuff_documents_chain
    )

    # LCEL 체인은 HumanMessage/AIMessage 객체를 사용하므로, 변환해줍니다.
    converted_chat_history = []
    for msg in chat_history:
        if msg['role'] == 'user':
            converted_chat_history.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            converted_chat_history.append(AIMessage(content=msg['content']))

    result = rag_chain.invoke(input={"input": query, "chat_history": converted_chat_history})

    # Streamlit 앱에서 사용하기 좋은 형식으로 반환값을 맞추어 줍니다.
    # (원래 코드에서 키 이름만 살짝 변경)
    return {
        "query": result.get('input'),
        "result": result.get('answer'),
        "source_documents": result.get('context', []),
    }


# core.py 파일의 가장 마지막 부분에 이 코드를 사용하세요.

if __name__ == "__main__":
    print("--- [테스트 시작] run_llm 함수를 직접 호출하여 테스트합니다. ---")

    # 1. 테스트용 채팅 기록 (처음에는 비어있음)
    chat_history_for_test = []

    # 2. 실제 사용자가 할 법한 한글 질문
    korean_query = "의료기기 사이버보안 가이던스에 대해 알려줘."

    print(f"\n[1차 질문] '{korean_query}'")

    # 3. run_llm 함수 호출
    result1 = run_llm(query=korean_query, chat_history=chat_history_for_test)

    # 4. 결과 출력
    print("\n[답변 1]")
    print(result1.get("result"))
    print("\n[참고 문서 1]")
    if result1.get("source_documents"):
        for doc in result1["source_documents"]:
            title = doc.metadata.get('title', 'N/A')
            page = doc.metadata.get('page', 'N/A')
            print(f"- {title} (페이지: {page})")

    print("\n" + "="*50)

    # 5. 후속 질문 테스트 (대화 기록 사용)
    #    이전 대화 내용을 채팅 기록에 추가합니다.
    chat_history_for_test.append({"role": "user", "content": korean_query})
    chat_history_for_test.append({"role": "assistant", "content": result1.get("result")})

    follow_up_query = "그럼 '취약한 보안(Weak security)'의 예시는 뭐야?"
    print(f"\n[2차 질문] (대화 기록 포함) '{follow_up_query}'")

    # 6. 대화 기록과 함께 run_llm 함수 다시 호출
    result2 = run_llm(query=follow_up_query, chat_history=chat_history_for_test)

    # 7. 두 번째 결과 출력
    print("\n[답변 2]")
    print(result2.get("result"))
    print("\n[참고 문서 2]")
    if result2.get("source_documents"):
        for doc in result2["source_documents"]:
            title = doc.metadata.get('title', 'N/A')
            page = doc.metadata.get('page', 'N/A')
            print(f"- {title} (페이지: {page})")