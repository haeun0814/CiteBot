import os
import json
import requests
from collections import Counter
from google import genai
from google.genai.types import HttpOptions

# ================= GCP GenAI 설정 =================
SERVICE_ACCOUNT_FILE = r"C:\Users\sdyha\OneDrive\바탕 화면\cohesive-sign-480005-i5-71e6afe89cd6.json"
PROJECT_ID = "cohesive-sign-480005-i5"
MODEL = "gemini-2.5-flash"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location="global",
    http_options=HttpOptions(api_version="v1")
)


def gcp_keyword_extract(text: str):
    """여러 abstracts에서 핵심 키워드 10개 추출"""
    try:
        prompt = (
            "Extract 10 important research keywords from this combined text. "
            "Return only a comma-separated list of keywords:\n\n"
            + text
        )
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )
        if hasattr(response, "text") and response.text:
            return [kw.strip() for kw in response.text.split(",")]
        return []
    except:
        return []


# ================= Semantic Scholar API 설정 =================
HEADERS = {"x-api-key": "F2EnQ7g01naKXhdeY3NZBgMCQP2BG0n2C0RDBRT8"}
API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def fetch_abstract_by_title(title: str):
    """제목으로 논문 정보 검색 → abstract 가져오기"""
    params = {
        "query": title,
        "fields": "title,abstract,authors,year,venue,paperId",
        "limit": 1,
    }
    r = requests.get(API_URL, params=params, headers=HEADERS)
    data = r.json()

    if "data" not in data or len(data["data"]) == 0:
        print(f"[경고] 논문을 찾을 수 없음: {title}")
        return None

    paper = data["data"][0]
    return {
        "title": paper.get("title"),
        "abstract": paper.get("abstract", ""),
        "paperId": paper.get("paperId"),
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "authors": [a.get("name") for a in paper.get("authors", [])],
    }


def extract_user_interest_keywords(papers):
    """여러 논문의 abstract를 합쳐서 키워드 추출"""
    combined_text = "\n\n".join([p["abstract"] for p in papers if p["abstract"]])
    if not combined_text.strip():
        return []

    keywords = gcp_keyword_extract(combined_text)
    return keywords


# ================= 메인 기능 =================
def process_user_input_titles(title_list):
    # 1. 제목 리스트로 abstracts 가져오기
    papers = []
    for t in title_list:
        paper = fetch_abstract_by_title(t)
        if paper:
            papers.append(paper)

    if not papers:
        print("가져올 논문이 없습니다.")
        return []

    # 2. 키워드 분석
    interest_keywords = extract_user_interest_keywords(papers)

    # 3. 결과 출력
    print("\n=== 가져온 논문 ===")
    for p in papers:
        print(f"- {p['title']} ({p['year']})")

    print("\n=== 추출된 관심사 키워드 ===")
    print(interest_keywords)

    return interest_keywords


# ================= 사용 예시 =================
if __name__ == "__main__":
    titles = [
        "Leveraging Large Language Models for Active Merchant Non-player Characters",
        "Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Game",
    ]

    process_user_input_titles(titles)
