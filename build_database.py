import json
import os
from typing import Any, Dict, List
import numpy as np
from google.cloud import aiplatform
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
from tqdm import tqdm
import faiss
import time


# ----------------------------
# 파일 로딩
# ----------------------------
print("[INFO] 논문 태그 데이터를 로드합니다...")
with open("citing_papers.json", "r", encoding="utf-8") as f:
    data = json.load(f)
paper_DATA = data

# ----------------------------
# GCP Vertex AI 설정
# ----------------------------

EMB_MODEL = "text-embedding-004" # Google Vertex AI 모델
EMB_DIM = 768                         # text-embedding-004 모델의 차원

# ----------------------------
# 임베딩 생성 클래스
# ----------------------------
class VertexAIEmbedder:
    """Vertex AI 임베딩 생성을 위한 클래스 (Retry 로직 추가됨)."""
    def __init__(self, model_name: str = EMB_MODEL):
        self.model_name = model_name
        self.model = TextEmbeddingModel.from_pretrained(self.model_name)
        self.dim = EMB_DIM

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """주어진 텍스트 목록에 대한 임베딩을 생성합니다."""
        # 배치 사이즈를 20에서 10으로 줄이는 것을 추천합니다 (요청 빈도 조절)
        if batch_size > 250:
            batch_size = 250
            
        if not texts:
            return np.empty((0, self.dim), dtype="float32")

        all_embeddings = []
        
        # tqdm 진행바
        for i in tqdm(range(0, len(texts), batch_size), desc="논문 데이터 임베딩 생성 중"):
            batch = texts[i:i + batch_size]
            
            # --- 재시도(Retry) 로직 시작 ---
            max_retries = 5       # 최대 5번까지 재시도
            retry_delay = 10      # 첫 대기 시간 10초
            success = False
            
            for attempt in range(max_retries):
                try:
                    # Vertex AI SDK 호출
                    resp = self.model.get_embeddings(batch)
                    embs = [np.array(d.values, dtype="float32") for d in resp]
                    all_embeddings.append(np.vstack(embs))
                    success = True
                    
                    # 성공했다면 API 부하를 줄이기 위해 짧게 대기 (선택사항)
                    time.sleep(1) 
                    break # 성공했으므로 재시도 루프 탈출
                    
                except Exception as e:
                    error_msg = str(e)
                    # 429 Quota 에러인 경우
                    if "429" in error_msg or "Quota exceeded" in error_msg:
                        print(f"\n[대기] 호출 한도 초과. {retry_delay}초 대기 후 재시도합니다... (시도 {attempt+1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2 # 대기 시간을 2배로 늘림 (Exponential Backoff)
                    else:
                        print(f"\n[오류] 알 수 없는 오류 발생: {e}")
                        break # 429 외의 에러는 중단
            
            # 재시도 해도 실패했다면, 데이터 정합성을 위해 전체 프로세스를 멈추는 게 안전합니다.
            if not success:
                raise RuntimeError(f"배치 처리 실패 (인덱스 {i}~{i+len(batch)}). 데이터 누락 방지를 위해 중단합니다.")
            # --- 재시도 로직 끝 ---

        if not all_embeddings:
            return np.empty((0, self.dim), dtype="float32")
            
        return np.vstack(all_embeddings)
# class VertexAIEmbedder:
#     """Vertex AI 임베딩 생성을 위한 클래스."""
#     def __init__(self, model_name: str = EMB_MODEL):
#         self.model_name = model_name
#         self.model = TextEmbeddingModel.from_pretrained(self.model_name)
#         self.dim = EMB_DIM

#     def encode(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
#         """주어진 텍스트 목록에 대한 임베딩을 생성합니다."""
#         # Vertex AI 'gecko' 모델의 최대 배치 크기는 250입니다.
#         if batch_size > 250:
#             print(f"[경고] 배치 크기({batch_size})가 Vertex AI 최대치(250)보다 큽니다. 250으로 조정합니다.")
#             batch_size = 250
            
#         if not texts:
#             return np.empty((0, self.dim), dtype="float32")

#         all_embeddings = []
#         for i in tqdm(range(0, len(texts), batch_size), desc="논문 데이터 임베딩 생성 중 (Vertex AI)"):
#             batch = texts[i:i + batch_size]
#             try:
#                 # Vertex AI SDK를 사용하여 임베딩 생성
#                 resp = self.model.get_embeddings(batch)
#                 embs = [np.array(d.values, dtype="float32") for d in resp]
#                 all_embeddings.append(np.vstack(embs))
#             except Exception as e:
#                 print(f"[오류] API 호출 중 오류 발생: {e}")
#                 # 특정 배치에서 오류가 나도 다음 배치를 시도하도록 continue
#                 continue
        
#         if not all_embeddings:
#             return np.empty((0, self.dim), dtype="float32")
            
#         return np.vstack(all_embeddings)

# ----------------------------
# 텍스트 전처리 유틸
# ----------------------------
def format_record_text(model: str, info: Dict[str, Any], max_field_len: int = 500, max_total_len: int = 4000) -> str:
    # """한 논문 딕셔너리를 key:value 형태로 모두 이어붙이되 길이를 제한해 토큰 초과를 방지합니다."""
    # parts = [f"model: {model}"]
    # for key, value in info.items():
    #     if value is None:
    #         continue
    #     if isinstance(value, list):
    #         value_str = ", ".join(map(str, value))
    #     else:
    #         value_str = str(value)
    #     if len(value_str) > max_field_len:
    #         value_str = value_str[:max_field_len] + "...(truncated)"
    #     parts.append(f"{key}: {value_str}")

    # text = " | ".join(parts)
    # if len(text) > max_total_len:
    #         text = text[:max_total_len] + "...(truncated)"
    # return text
    
    """한 논문 딕셔너리를 key:value 형태로 모두 이어붙입니다. (길이 제한 없음)"""  
    parts = [f"model: {model}"]

    for key, value in info.items():
        if value is None:
            continue
        # 리스트는 문자열로 합침
        if isinstance(value, list):
            value_str = ", ".join(map(str, value))
        else:
            value_str = str(value)
        parts.append(f"{key}: {value_str}")
    # 모든 key:value를 " | "로 이어붙임
    text = " | ".join(parts)
    return text

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
        # 모든 필드를 포함하되 필드/전체 길이를 제한해 토큰 초과를 방지합니다.
        text = format_record_text(model, info)
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
    print(f"[INFO] 인덱스 빌드 완료 (ntotal={index.ntotal})")
    
    # 4. 인덱스 및 메타데이터 저장
    paper_PATH = "citing_papers.faiss"
    paper_MAP_PATH = "citing_papers_to_model.json"

    print(f"[INFO] FAISS 인덱스를 '{paper_PATH}' 파일로 저장합니다...")
    faiss.write_index(index, paper_PATH)
    
    print(f"[INFO] 인덱스-모델명 매핑 데이터를 '{paper_MAP_PATH}' 파일로 저장합니다...")
    index_to_model_map = {i: model for i, model in enumerate(paper_models)}
    with open(paper_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(index_to_model_map, f, ensure_ascii=False, indent=2)
        
    print("\n[성공] 논문 임베딩 데이터베이스 구축이 완료되었습니다!")
    print(f"-> FAISS 인덱스: {paper_PATH}")
    print(f"-> 인덱스 맵핑: {paper_MAP_PATH} (entries={len(index_to_model_map)})")

# ----------------------------
# 스크립트 실행
# ----------------------------
if __name__ == '__main__':
    build_paper_embedding_database()
