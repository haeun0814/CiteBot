import requests

def get_citing_papers(arxiv_id):
    # 1. 먼저 논문의 내부 ID 가져오기
    paper_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=paperId,title"
    r = requests.get(paper_url)
    paper = r.json()
    paper_id = paper.get("paperId")
    if not paper_id:
        print("논문을 찾을 수 없습니다.")
        return []

    # 2. 인용 논문(citations) 전용 endpoint 호출
    citations_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields=title,authors,externalIds"
    r2 = requests.get(citations_url)
    data = r2.json()

    if "data" not in data or not data["data"]:
        print("인용 데이터가 없습니다.")
        return []

    citing_papers = []
    for c in data["data"]:
        cited = c["citingPaper"]
        citing_papers.append({
            "title": cited["title"],
            "authors": [a["name"] for a in cited["authors"]],
            "arxiv_id": cited.get("externalIds", {}).get("ArXiv")
        })

    return citing_papers

# 테스트
citing = get_citing_papers("2412.11189")
print(f"총 {len(citing)}편 인용 논문 발견")
for i, p in enumerate(citing[:5], 1):
    print(f"{i}. {p['title']}")
