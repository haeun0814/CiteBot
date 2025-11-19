import requests
import json
import os

API_KEY = "F2EnQ7g01naKXhdeY3NZBgMCQP2BG0n2C0RDBRT8"
HEADERS = {"x-api-key": API_KEY}

def save_citing_papers_json_single(arxiv_id, min_citations=0, max_year=2025, save_path="citing_papers.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    paper_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=paperId,title"
    r = requests.get(paper_url, headers=HEADERS)
    paper = r.json()
    paper_id = paper.get("paperId")
    if not paper_id:
        print("논문을 찾을 수 없습니다.")
        return

    fields = [
        "title",
        "authors",
        "externalIds",
        "venue",
        "year",
        "publicationTypes",
        "citationCount",
        "fieldsOfStudy"
    ]
    
    citing_papers = []
    offset = 0
    limit = 100
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
                "title": cited["title"],
                "authors": [a["name"] for a in cited["authors"]],
                "arxiv_id": cited.get("externalIds", {}).get("ArXiv"),
                "venue": cited.get("venue"),
                "year": cited.get("year"),
                "publicationTypes": cited.get("publicationTypes"),
                "citationCount": cited.get("citationCount"),
                "fieldsOfStudy": cited.get("fieldsOfStudy")
            })

        offset += limit
        if len(data["data"]) < limit:
            break

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(citing_papers, f, ensure_ascii=False, indent=4)

    print(f"총 {len(citing_papers)}개의 인용 논문이 '{save_path}' 파일에 저장되었습니다.")


save_citing_papers_json_single("논문 id", save_path=r"파일 경로")
# 추후에 논문 id는 입력 받게 수정 필요
