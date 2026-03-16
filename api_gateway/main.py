"""
API Gateway — nhận user_id, trả recommendations.
Đây là serving code, chạy 24/7 trong EKS.

Flow:
  1. Nhận GET /infer?user_id=xxx
  2. Check warm/cold user
  3. Warm: query Redis cho similar items + FAISS candidates
  4. Cold: query Redis cho popular items
  5. Load features từ Feature Store
  6. Score với ranking model
  7. Trả top-K recommendations
"""

from fastapi import FastAPI, HTTPException
import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title="RecSys RetailRocket API")

# Globals — loaded once at startup
MODEL = None
USER2ID = None
POPULAR_ITEMS = None


@app.on_event("startup")
async def load_artifacts():
    """Load model + artifacts khi server start.
    Artifacts được download từ S3/MLflow trước khi server start
    (bởi init container trong K8s, hoặc startup script).
    """
    global MODEL, USER2ID, POPULAR_ITEMS
    
    artifact_dir = os.getenv("ARTIFACT_DIR", "/app/artifacts")
    
    try:
        import lightgbm as lgb
        MODEL = lgb.Booster(model_file=f"{artifact_dir}/lgbm_ranker.txt")
        logger.info("Model loaded")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    try:
        with open(f"{artifact_dir}/user2id.json") as f:
            USER2ID = json.load(f)
        with open(f"{artifact_dir}/popular_items.json") as f:
            POPULAR_ITEMS = json.load(f)
        logger.info(f"Artifacts loaded: {len(USER2ID)} users, {len(POPULAR_ITEMS)} popular items")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")


@app.get("/health")
async def health():
    """Health check — K8s liveness probe gọi endpoint này."""
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/infer")
async def infer(user_id: str, k: int = 20):
    """Main inference endpoint.
    
    Args:
        user_id: visitor ID
        k: number of recommendations to return
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check warm/cold
    is_warm = user_id in USER2ID
    
    if is_warm:
        # TODO: query Redis cho personalized candidates
        # TODO: query Feature Store cho user features  
        # TODO: score với ranking model
        candidates = POPULAR_ITEMS[:k]  # placeholder
    else:
        # Cold user: chỉ popular items
        candidates = POPULAR_ITEMS[:k]
    
    return {
        "user_id": user_id,
        "is_warm_user": is_warm,
        "recommendations": candidates,
        "n_recommendations": len(candidates),
    }# initial build
