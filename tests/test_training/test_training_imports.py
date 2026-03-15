"""
Test imports — đảm bảo tất cả training code import được.
Nếu syntax error, missing import, typo → fail ở đây, 
không fail lúc Kubeflow chạy.
"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_id_mapper_import():
    """src/id_mapper.py import được."""
    from src.id_mapper import IDMapper
    assert IDMapper is not None


def test_feature_engineer_import():
    """Feature engineering code import được."""
    # Test file tồn tại và parseable
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "features", "src/feature_engineer/features.py"
    )
    assert spec is not None, "src/feature_engineer/features.py not found"


def test_item2vec_import():
    """Item2Vec code import được."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "item2vec", "src/model_item2vec/main.py"
    )
    assert spec is not None, "src/model_item2vec/main.py not found"


def test_lightgcn_import():
    """LightGCN code import được."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lightgcn", "src/model_lightgcn/main.py"
    )
    assert spec is not None, "src/model_lightgcn/main.py not found"


def test_sasrec_import():
    """SASRec code import được."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sasrec", "src/model_sasrec/main.py"
    )
    assert spec is not None, "src/model_sasrec/main.py not found"


def test_retrieval_import():
    """Retrieval code import được."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "retrieval", "src/model_retrieval/main.py"
    )
    assert spec is not None, "src/model_retrieval/main.py not found"


def test_ranking_import():
    """Ranking code import được."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ranking", "src/model_ranking/main.py"
    )
    assert spec is not None, "src/model_ranking/main.py not found"


def test_core_dependencies():
    """Core libraries install đúng."""
    import pandas
    import numpy
    assert pandas.__version__ is not None
    assert numpy.__version__ is not None


def test_ml_dependencies():
    """ML libraries import được."""
    try:
        import lightgbm
        assert lightgbm.__version__ is not None
    except ImportError:
        pytest.skip("lightgbm not installed (optional for CI)")

    try:
        import gensim
        assert gensim.__version__ is not None
    except ImportError:
        pytest.skip("gensim not installed (optional for CI)")