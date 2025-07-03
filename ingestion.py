from dotenv import load_dotenv
load_dotenv()

import os
import re
from firecrawl import FirecrawlApp
from urllib.parse import urljoin
from MDCG_web_scraper import parse_md_table, filter_documents

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
index_name = "medical-ai-regulation"
fc_api_key = os.getenv("FIRECRAWL_API_KEY")

def ingest_docs():
    loader = ReadTheDocsLoader("C:\\Users\\tada2\\OneDrive\\문서\\LangChain\\documentation-helper\\langchain-docs\\api.python.langchain.com\\en\\latest", encoding="utf-8")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(documents=raw_documents)
    for doc in documents:
        new_url = doc.metadata['source']
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    vector_store = PineconeVectorStore.from_existing_index(
        embedding=embeddings, index_name="langchain-doc-index"
    )
    batch_size = 100 # 한 번에 업로드할 문서 청크 수 (조절 가능)
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(batch) # 기존 vector_store 객체에 문서를 추가
        print(f"Batch {i // batch_size + 1} added.")


def scrape_MDCG_pdf() -> list:

    app = FirecrawlApp(api_key=fc_api_key)
    scrape_result = app.scrape_url(
        'https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en',
        )
    print("스크래핑 성공!")
    all_documents = parse_md_table(scrape_result.markdown)
    print(f"파싱 완료: {len(all_documents)}개의 문서 정보 추출")

    final_keywords = [
    'artificial intelligence',
    'aia',
    'software',
    'mdsw',
    'cybersecurity',
    'algorithm',]

    #  데이터 필터링 (새로 만든 함수 호출)
    final_documents = filter_documents(all_documents, final_keywords)
    print(f"필터링 완료: 최종 {len(final_documents)}개의 문서 선택됨")

    return final_documents


def ingest_pdfs_to_pinecone(documents_with_links: list, namespace: str) -> None:
    """
    2단계 & 3단계: PDF 링크 리스트를 받아, 내용을 처리하고
    '배치(batch)'로 나누어 Pinecone에 저장합니다.
    PDF 링크 리스트와 네임스페이스를 받아, 내용을 처리하고
    지정된 네임스페이스에 배치로 나누어 Pinecone에 저장합니다.
    """
    if not documents_with_links:
        print("처리할 문서가 없습니다. 작업을 종료합니다.")
        return

    print(f"\n2단계: 총 {len(documents_with_links)}개의 PDF 문서 처리를 시작합니다...")

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    base_url = "https://health.ec.europa.eu"

    for i, doc_info in enumerate(documents_with_links, 1):
        pdf_url = doc_info.get('url')

        # URL이 없거나 PDF 링크가 아니면 건너뜁니다.
        if not pdf_url or not pdf_url.endswith('.pdf'):
            continue

        # 상대 경로를 절대 경로로 변환
        full_pdf_url = urljoin(base_url, pdf_url)
        print(f"  ({i}/{len(documents_with_links)}) 처리 중: {full_pdf_url}")

        try:
            loader = PyPDFLoader(full_pdf_url)
            pdf_pages = loader.load()

            split_docs = text_splitter.split_documents(pdf_pages)

            # **핵심:** 각 청크에 원본 문서의 메타데이터(참조, 제목)를 추가해줍니다.
            # 이렇게 하면 나중에 어떤 문서에서 온 정보인지 추적할 수 있습니다.
            for chunk in split_docs:
                chunk.metadata['data_source'] = namespace # 'mdcg' 또는 'fda' 같은 출처 이름 추가
                chunk.metadata['original_source'] = full_pdf_url
                chunk.metadata['reference'] = doc_info.get('reference', 'N/A')
                chunk.metadata['title'] = doc_info.get('title', 'N/A')

            all_chunks.extend(split_docs)
            print(f"    -> {len(split_docs)}개의 청크 생성 완료.")
        except Exception as e:
            print(f"    -> 에러 발생, 이 PDF를 건너뜁니다: {e}")

    if not all_chunks:
        print("\n처리 가능한 문서 청크가 없습니다. 작업을 종료합니다.")
        return

    print(f"\n3단계: 총 {len(all_chunks)}개의 청크를 Pinecone에 저장합니다...")

    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    # 데이터를 보낼 배치(묶음) 크기 설정
    batch_size = 100
    # 전체 청크를 배치 크기만큼 잘라서 반복문 실행
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        print(f"  {i+1}번부터 {min(i + batch_size, len(all_chunks))}번까지의 청크를 업로드 중...")
        vector_store.add_documents(batch, namespace='mdcg')

    print("\n미션 완료! 필터링된 PDF의 내용이 성공적으로 Pinecone에 저장되었습니다.")

if __name__ == "__main__":
    final_documents = scrape_MDCG_pdf()
    ingest_pdfs_to_pinecone(final_documents, namespace="mdcg")

    # 나중에 FDA 데이터를 처리할 때는 이렇게 호출하면 됩니다.
    # fda_documents = scrape_FDA_pdf() # FDA용 스크래핑 함수 (미래에 만들 것)
    # ingest_pdfs_to_pinecone(fda_documents, namespace="fda")