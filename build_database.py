import json
import os
from typing import Any, Dict, List
import numpy as np
from google.cloud import aiplatform
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
from tqdm import tqdm
import faiss


# ----------------------------
# 파일 로딩
# ----------------------------
print("[INFO] 논문 태그 데이터를 로드합니다...")
with open('paper.json', 'r', encoding='utf-8') as f:
    paper_DATA: Dict[str, Dict[str, Any]] = json.load(f)

# ----------------------------
# GCP Vertex AI 설정
# ----------------------------

EMB_MODEL = "text-embedding-004" # Google Vertex AI 모델
EMB_DIM = 768                         # text-embedding-004 모델의 차원

# ----------------------------
# 임베딩 생성 클래스
# ----------------------------
class VertexAIEmbedder:
    """Vertex AI 임베딩 생성을 위한 클래스."""
    def __init__(self, model_name: str = EMB_MODEL):
        self.model_name = model_name
        self.model = TextEmbeddingModel.from_pretrained(self.model_name)
        self.dim = EMB_DIM

    def encode(self, texts: List[str], batch_size: int = 250) -> np.ndarray:
        """주어진 텍스트 목록에 대한 임베딩을 생성합니다."""
        # Vertex AI 'gecko' 모델의 최대 배치 크기는 250입니다.
        if batch_size > 250:
            print(f"[경고] 배치 크기({batch_size})가 Vertex AI 최대치(250)보다 큽니다. 250으로 조정합니다.")
            batch_size = 250
            
        if not texts:
            return np.empty((0, self.dim), dtype="float32")

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="논문 데이터 임베딩 생성 중 (Vertex AI)"):
            batch = texts[i:i + batch_size]
            try:
                # Vertex AI SDK를 사용하여 임베딩 생성
                resp = self.model.get_embeddings(batch)
                embs = [np.array(d.values, dtype="float32") for d in resp]
                all_embeddings.append(np.vstack(embs))
            except Exception as e:
                print(f"[오류] API 호출 중 오류 발생: {e}")
                # 특정 배치에서 오류가 나도 다음 배치를 시도하도록 continue
                continue
        
        if not all_embeddings:
            return np.empty((0, self.dim), dtype="float32")
            
        return np.vstack(all_embeddings)


# ----------------------------
# 데이터베이스 구축 메인 로직
# ----------------------------
def build_paper_embedding_database():
    """논문 데이터에 대한 FAISS 임베딩 데이터베이스를 구축하고 저장합니다."""
    
    # 1. 논문 데이터를 텍스트로 변환
    print("[INFO] 논문 데이터를 텍스트 형식으로 변환합니다...")
    paper_models = list(paper_DATA.keys())
    paper_texts = []
    for model in paper_models:
        info = paper_DATA[model]
        text = f"모델명: {model}. " + " ".join(
            f"{key}: {value}" for key, value in info.items() if value is not None
        )
        paper_texts.append(text)
        
    # 2. 텍스트 임베딩 생성
    print("[INFO] OpenAI 임베딩을 생성합니다...")
    embedder = VertexAIEmbedder()
    paper_embeddings = embedder.encode(paper_texts)
    
    if paper_embeddings.size == 0:
        print("[오류] 임베딩 생성에 실패하여 데이터베이스를 구축할 수 없습니다.")
        return

    # 3. FAISS 인덱스 구축
    print(f"[INFO] {paper_embeddings.shape[0]}개의 임베딩으로 FAISS 인덱스를 구축합니다...")
    index = faiss.IndexFlatL2(EMB_DIM)
    index.add(paper_embeddings)
    
    # 4. 인덱스 및 메타데이터 저장
    paper_PATH = "paper.faiss"
    paper_MAP_PATH = "paper_to_model.json"

    print(f"[INFO] FAISS 인덱스를 '{paper_PATH}' 파일로 저장합니다...")
    faiss.write_index(index, paper_PATH)
    
    print(f"[INFO] 인덱스-모델명 매핑 데이터를 '{paper_MAP_PATH}' 파일로 저장합니다...")
    index_to_model_map = {i: model for i, model in enumerate(paper_models)}
    with open(paper_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(index_to_model_map, f, ensure_ascii=False, indent=2)
        
    print("\n[성공] 논문 임베딩 데이터베이스 구축이 완료되었습니다!")
    print(f"-> FAISS 인덱스: {paper_PATH}")
    print(f"-> 인덱스 맵핑: {paper_MAP_PATH}")

# ----------------------------
# 스크립트 실행
# ----------------------------
if __name__ == '__main__':
    build_paper_embedding_database()