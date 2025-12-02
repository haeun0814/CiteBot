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
client = genai.Client(http_options=HttpOptions(api_version="v1"))

def gcp_summarize_text(text: str):
    if not text.strip():
        return "[요약 없음]"
    try:
        prompt = f"summarize: {text}"
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,  # 문자열로 전달 (response.text 사용 가능)
        )
        # 우선 response.text 확인
        if hasattr(response, "text") and response.text and response.text.strip():
            return response.text.strip()
        # candidates 구조 확인
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

def save_citing_papers_json_single(arxiv_id: str, min_citations=0, max_year=2025,
                                   save_path="citing_papers.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 원 논문 ID 가져오기
    paper_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=paperId,title"
    r = requests.get(paper_url, headers=HEADERS)
    paper = r.json()
    paper_id = paper.get("paperId")
    if not paper_id:
        print("논문을 찾을 수 없습니다.")
        return

    fields = [
        "title","authors","externalIds","venue","year",
        "publicationTypes","citationCount","fieldsOfStudy",
        "abstract","isOpenAccess","url"
    ]

    citing_papers = []
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

            citing_papers.append({
                "title": cited.get("title"),
                "authors": [a["name"] for a in cited.get("authors", [])],
                "arxiv_id": cited.get("externalIds", {}).get("ArXiv"),
                "venue": cited.get("venue"),
                "year": cited.get("year"),
                "publicationTypes": cited.get("publicationTypes"),
                "citationCount": cited.get("citationCount"),
                "fieldsOfStudy": cited.get("fieldsOfStudy"),
                "abstract": cited.get("abstract", ""),
                "abstract_summary_gcp": "[요약 없음]",  # 요약은 나중에 업데이트
                "url": cited.get("url"),
                "isOpenAccess": cited.get("isOpenAccess")
            })

        offset += limit
        if len(data["data"]) < limit:
            break

    # 기존 파일 읽기
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # 중복 제거
    updated_data = existing_data + citing_papers
    seen = set()
    unique_data = []
    for paper in updated_data:
        key = paper.get("arxiv_id") or paper.get("title")
        if key not in seen:
            unique_data.append(paper)
            seen.add(key)

    # 저장
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=4)

    print(f"총 {len(unique_data)}개의 인용 논문이 '{save_path}' 파일에 저장/업데이트되었습니다.")


# ================= JSON 불러와서 요약 업데이트 =================
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

    updated_papers = []
    for paper in papers:
        abstract_text = paper.get("abstract", "")
        if abstract_text and (not paper.get("abstract_summary_gcp") or paper["abstract_summary_gcp"] in ["[요약 없음]", "[요약 실패]"]):
            summary = gcp_summarize_text(abstract_text)
            paper["abstract_summary_gcp"] = summary
        updated_papers.append(paper)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(updated_papers, f, ensure_ascii=False, indent=4)

    print(f"총 {len(updated_papers)}개의 논문 요약이 '{json_path}'에 업데이트되었습니다.")


# ================= 사용 예시 =================
if __name__ == "__main__":
    json_file = r"C:\Users\sdyha\OneDrive\문서\GitHub\CiteBot\citing_papers.json"

    # 1단계: 인용 논문 저장
    save_citing_papers_json_single("1706.03762", save_path=json_file)

    # 2단계: 저장된 JSON 불러와서 요약 업데이트
    update_json_with_summaries(json_file)
