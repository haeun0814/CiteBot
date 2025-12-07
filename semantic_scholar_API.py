import os
import json
import requests
from google import genai
from google.genai.types import HttpOptions

# ================= GCP GenAI 설정 =================
SERVICE_ACCOUNT_FILE = r"C:\Users\sdyha\OneDrive\바탕 화면\cohesive-sign-480005-i5-71e6afe89cd6.json"
PROJECT_ID = "cohesive-sign-480005-i5"
MODEL = "gemini-2.5-flash"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE

client = genai.Client(
    vertexai=True,
    project="cohesive-sign-480005-i5",
    location="us-central1"
)


# ================= GCP 요약 함수 =================
def gcp_summarize_text(text: str):
    if not text.strip():
        return "[요약 없음]"
    try:
        prompt = f"summarize: {text}"
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )

        if hasattr(response, "text") and response.text:
            return response.text.strip()

        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()

        return "[요약 실패]"
    except Exception as e:
        print("Gemini 요약 에러:", e)
        return "[요약 실패]"



# ================= Semantic Scholar API 설정 =================
HEADERS = {"x-api-key": "F2EnQ7g01naKXhdeY3NZBgMCQP2BG0n2C0RDBRT8"}


# ================= (추가) 제목으로 paperId 찾기 =================
def get_paperId_from_title(title: str):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": 1,
        "fields": "paperId,title"
    }

    r = requests.get(url, params=params, headers=HEADERS)
    data = r.json()

    if "data" not in data or not data["data"]:
        print(f"[오류] 제목으로 논문을 찾을 수 없음: {title}")
        return None

    paper_id = data["data"][0]["paperId"]
    print(f"[검색 성공] '{title}' → paperId = {paper_id}")
    return paper_id



# ================= 인용 논문 저장 (dict 구조) =================
def save_citing_papers_json_single_by_title(title: str, min_citations=0, max_year=2025,
                                            save_path="citing_papers.json"):

    # 1) 제목으로 paperId 찾기
    paper_id = get_paperId_from_title(title)
    if paper_id is None:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 원 논문 정보 요청
    paper_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=paperId,title,abstract,authors,year,venue,fieldsOfStudy,isOpenAccess,url"
    r = requests.get(paper_url, headers=HEADERS)
    paper = r.json()

    fields = [
        "title", "authors", "externalIds", "venue", "year",
        "publicationTypes", "citationCount", "fieldsOfStudy",
        "abstract", "isOpenAccess", "url"
    ]

    citing_papers_dict = {}

    # 원 논문 포함
    citing_papers_dict[title] = {
        "title": paper.get("title"),
        "authors": [a["name"] for a in paper.get("authors", [])],
        "arxiv_id": paper.get("externalIds", {}).get("ArXiv"),
        "venue": paper.get("venue"),
        "year": paper.get("year"),
        "publicationTypes": None,
        "citationCount": None,
        "fieldsOfStudy": paper.get("fieldsOfStudy"),
        "abstract": paper.get("abstract", ""),
        "abstract_summary_gcp": "[요약 없음]",
        "url": paper.get("url"),
        "isOpenAccess": paper.get("isOpenAccess")
    }

    # ========== 인용 논문 수집 ==========

    offset = 0
    limit = 10

    while True:
        citations_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields={','.join(fields)}&limit={limit}&offset={offset}"
        r2 = requests.get(citations_url, headers=HEADERS)
        data = r2.json()

        if "data" not in data or not data["data"]:
            break

        for c in data["data"]:
            cited = c["citingPaper"]

            if cited.get("citationCount", 0) < min_citations:
                continue
            if cited.get("year") is None or cited.get("year") > max_year:
                continue

            key = cited.get("externalIds", {}).get("ArXiv") or cited.get("title")

            citing_papers_dict[key] = {
                "title": cited.get("title"),
                "authors": [a["name"] for a in cited.get("authors", [])],
                "arxiv_id": cited.get("externalIds", {}).get("ArXiv"),
                "venue": cited.get("venue"),
                "year": cited.get("year"),
                "publicationTypes": cited.get("publicationTypes"),
                "citationCount": cited.get("citationCount"),
                "fieldsOfStudy": cited.get("fieldsOfStudy"),
                "abstract": cited.get("abstract", ""),
                "abstract_summary_gcp": "[요약 없음]",
                "url": cited.get("url"),
                "isOpenAccess": cited.get("isOpenAccess")
            }

        offset += limit
        if len(data["data"]) < limit:
            break

    # JSON 병합/저장
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
    else:
        existing = {}

    existing.update(citing_papers_dict)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=4)

    print(f"총 {len(existing)}개의 논문이 '{save_path}'에 저장/업데이트되었습니다.")



# ================= JSON 요약 업데이트 =================
def update_json_with_summaries(json_path: str):
    if not os.path.exists(json_path):
        print("JSON 파일이 존재하지 않습니다:", json_path)
        return

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            papers = json.load(f)
        except json.JSONDecodeError:
            print("JSON 파일을 읽을 수 없습니다.")
            return

    for key, paper in papers.items():
        abstract_text = paper.get("abstract", "")
        summary_before = paper.get("abstract_summary_gcp")

        if abstract_text and (not summary_before or summary_before in ["[요약 없음]", "[요약 실패]"]):
            papers[key]["abstract_summary_gcp"] = gcp_summarize_text(abstract_text)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=4)

    print(f"총 {len(papers)}개의 논문 요약이 '{json_path}'에 업데이트되었습니다.")



# ================= 실행 예시 =================
if __name__ == "__main__":
    json_file = r"C:\Users\sdyha\OneDrive\문서\GitHub\CiteBot\citing_papers.json"

    # 제목으로 인용 논문 수집
    save_citing_papers_json_single_by_title(
        "Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Games",
        save_path=json_file
    )

    # 요약 업데이트
    update_json_with_summaries(json_file)
