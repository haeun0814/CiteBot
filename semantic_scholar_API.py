import os
import json
import requests
from google import genai
from google.genai.types import HttpOptions

# ================= GCP GenAI 설정 =================
SERVICE_ACCOUNT_FILE = r"C:\Users\sdyha\OneDrive\바탕 화면\cohesive-sign-480005-i5-71e6afe89cd6.json"
PROJECT_ID = "cohesive-sign-480005-i5"
MODEL = "gemini-2.5-flash"

# 환경 변수로 서비스 계정 등록
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE

# GenAI 클라이언트 생성
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

        if hasattr(response, "text") and response.text and response.text.strip():
            return response.text.strip()

        elif response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()

        else:
            print("Gemini 응답이 비어있습니다. 입력 텍스트 길이:", len(text))
            return "[요약 실패]"

    except Exception as e:
        print("Gemini 요약 에러:", e)
        return "[요약 실패]"



# ================= Semantic Scholar API 설정 =================
HEADERS = {"x-api-key": "F2EnQ7g01naKXhdeY3NZBgMCQP2BG0n2C0RDBRT8"}


# ================= 인용 논문 저장 (dict 구조) =================
def save_citing_papers_json_single(arxiv_id: str, min_citations=0, max_year=2025,
                                   save_path="citing_papers.json"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 원 논문 ID 가져오기
    paper_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=paperId,title,abstract,authors,year,venue,fieldsOfStudy,isOpenAccess,url"
    r = requests.get(paper_url, headers=HEADERS)
    paper = r.json()
    paper_id = paper.get("paperId")
    if not paper_id:
        print("논문을 찾을 수 없습니다.")
        return

    fields = [
        "title", "authors", "externalIds", "venue", "year",
        "publicationTypes", "citationCount", "fieldsOfStudy",
        "abstract", "isOpenAccess", "url"
    ]

    # dict 생성
    citing_papers_dict = {}

    # ========== 원 논문도 dict에 포함 ==========
    citing_papers_dict[arxiv_id] = {
        "title": paper.get("title"),
        "authors": [a["name"] for a in paper.get("authors", [])],
        "arxiv_id": arxiv_id,
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

            # key는 arxiv id 또는 title
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

    # 기존 JSON 불러오기
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # 병합
    existing_data.update(citing_papers_dict)

    # 저장
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"총 {len(existing_data)}개의 논문이 '{save_path}'에 저장/업데이트되었습니다.")



# ================= JSON 요약 업데이트 =================
def update_json_with_summaries(json_path: str):
    if not os.path.exists(json_path):
        print("JSON 파일이 존재하지 않습니다:", json_path)
        return

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            papers = json.load(f)    # dict 로 로딩
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

    # 1단계: 원 논문 + 인용 논문 dict 형태로 저장
    save_citing_papers_json_single("2411.10109", save_path=json_file)

    # 2단계: 요약 업데이트
    update_json_with_summaries(json_file)
