# 마크다운 파싱
import re

# 1. 마크다운 파싱 함수
def parse_md_table(markdown_text):
    """
    사용자가 지정한 함수명.
    마크다운 텍스트에서 테이블과 리스트를 파싱하여 문서 정보를 추출합니다.
    """
    documents = []

    # ----------------------------------------------------
    # 1. 마크다운 테이블 파싱
    # ----------------------------------------------------
    try:
        # 테이블이 포함된 'Current guidance' 섹션만 추출
        table_section = markdown_text.split('## Current guidance')[1].split('## Latest updates')[0]
    except IndexError:
        table_section = "" # 섹션이 없으면 비워둠

    for line in table_section.strip().split('\n'):
        if not line.startswith('|') or '---' in line or 'Reference' in line:
            continue

        columns = [col.strip() for col in line.split('|')][1:-1]
        if len(columns) < 3:
            continue

        reference_cell, title_cell = columns[0], columns[1]

        # 한 셀에 <br>로 여러 문서가 있는 경우를 처리
        reference_parts = reference_cell.split('<br>')
        title_parts = title_cell.split('<br>')

        for i in range(len(reference_parts)):
            ref_part = reference_parts[i].strip()
            title_part = title_parts[i].strip() if i < len(title_parts) else ''

            # 링크 패턴 [text](url)을 찾기 위한 정규식
            ref_match = re.search(r'\[(.*?)\]\((.*?)\)', ref_part)
            title_match = re.search(r'\[(.*?)\]\((.*?)\)', title_part)

            reference, title, url = "", "", None

            if ref_match: # Reference 셀에 링크가 있는 경우
                reference, url, title = ref_match.group(1), ref_match.group(2), title_part
            elif title_match: # Title 셀에 링크가 있는 경우
                reference, title, url = ref_part, title_match.group(1), title_match.group(2)
            else: # 링크가 없는 경우
                reference, title = ref_part, title_part

            if reference or title:
                documents.append({
                    'reference': reference, 'title': title, 'url': url
                })

    # ----------------------------------------------------
    # 2. 'Latest updates' 리스트 섹션 파싱
    # ----------------------------------------------------
    try:
        updates_section = markdown_text.split('## Latest updates')[1]
        # 리스트 형식의 링크를 모두 찾음
        update_matches = re.findall(r'\[(MDCG.*? \(.*?\))\]\((.*?)\)', updates_section)

        for match in update_matches:
            full_title, url = match[0], match[1]
            reference = full_title.split(' - ')[0]

            documents.append({
                'reference': reference, 'title': full_title, 'url': url
            })
    except IndexError:
        pass # 섹션이 없으면 무시

    return documents

# 2. 키워드 필터링 로직
def filter_documents(all_documents, keywords):
    """
    문서 목록에서 키워드가 포함된 문서를 필터링합니다.
    """
    final_documents = []
    for doc in all_documents:
        title_lower = doc['title'].lower()
        if any(keyword in title_lower for keyword in keywords):
            if doc not in final_documents:
                final_documents.append(doc)
    return final_documents
