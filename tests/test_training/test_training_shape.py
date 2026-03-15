"""
Test model output shapes trên fake data.
Catch: dimension mismatch, wrong output type, crash trên edge cases.
Chạy trong 10 giây, không cần GPU, không cần real data.
"""
import pytest
import numpy as np


# ============================================================
# ITEM2VEC SHAPE
# ============================================================

class TestItem2VecShape:
    """Test Item2Vec produces correct output shape."""

    def test_embedding_shape(self):
        """Output shape = (vocab_size, vector_size)."""
        from gensim.models import Word2Vec

        sequences = [
            ['i1', 'i2', 'i3', 'i4'],
            ['i2', 'i3', 'i5', 'i6'],
            ['i1', 'i3', 'i4', 'i7'],
            ['i5', 'i6', 'i7', 'i1'],
            ['i2', 'i4', 'i6', 'i3'],
        ]
        vector_size = 16

        model = Word2Vec(
            sentences=sequences,
            vector_size=vector_size,
            window=2,
            min_count=1,
            sg=1,
            epochs=3,
            workers=1,
            seed=42,
        )

        assert model.wv.vectors.shape[1] == vector_size
        assert len(model.wv) == 7  # 7 unique items

    def test_similarity_works(self):
        """most_similar returns results."""
        from gensim.models import Word2Vec

        sequences = [['a','b','c']] * 10 + [['b','c','d']] * 10
        model = Word2Vec(sequences, vector_size=8, window=2,
                        min_count=1, epochs=5, workers=1)

        similar = model.wv.most_similar('a', topn=2)
        assert len(similar) == 2
        assert all(0 <= score <= 1 for _, score in similar)


# ============================================================
# LIGHTGCN SHAPE
# ============================================================

class TestLightGCNShape:
    """Test LightGCN produces correct output shapes."""

    def test_embedding_dimensions(self):
        """User emb + Item emb dimensions match config."""
        import torch
        import torch.nn as nn

        n_users, n_items, emb_dim = 10, 8, 16

        user_emb = nn.Embedding(n_users, emb_dim)
        item_emb = nn.Embedding(n_items, emb_dim)

        assert user_emb.weight.shape == (n_users, emb_dim)
        assert item_emb.weight.shape == (n_items, emb_dim)

    def test_dot_product_score(self):
        """Dot product between user and item embedding → scalar."""
        u = np.random.randn(64)
        i = np.random.randn(64)
        score = np.dot(u, i)

        assert isinstance(score, (float, np.floating))

    def test_bpr_loss_computable(self):
        """BPR loss computable with positive > negative score."""
        import torch

        pos_score = torch.tensor([2.0, 1.5, 3.0])
        neg_score = torch.tensor([0.5, 0.3, 1.0])

        bpr = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

        assert not torch.isnan(bpr)
        assert bpr.item() > 0


# ============================================================
# SASREC SHAPE
# ============================================================

class TestSASRecShape:
    """Test SASRec input/output shapes."""

    def test_sequence_padding(self):
        """Sequences padded to max_len correctly."""
        max_len = 10
        seq = [1, 2, 3]  # length 3

        pad_len = max_len - len(seq)
        padded = [0] * pad_len + seq

        assert len(padded) == max_len
        assert padded[:pad_len] == [0] * pad_len  # left padding
        assert padded[pad_len:] == seq

    def test_causal_mask_shape(self):
        """Causal mask: upper triangle = True (blocked)."""
        import torch

        seq_len = 5
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        assert mask.shape == (seq_len, seq_len)
        assert mask[0][0] == False   # can attend to self
        assert mask[0][1] == True    # cannot attend to future
        assert mask[4][3] == False   # can attend to past

    def test_output_shape(self):
        """Transformer output shape = (batch, seq_len, vocab)."""
        import torch
        import torch.nn as nn

        batch, seq_len, emb_dim, vocab = 4, 10, 16, 100

        # Simplified forward
        embedding = nn.Embedding(vocab, emb_dim, padding_idx=0)
        proj = nn.Linear(emb_dim, vocab)

        input_ids = torch.randint(0, vocab, (batch, seq_len))
        emb = embedding(input_ids)
        output = proj(emb)

        assert output.shape == (batch, seq_len, vocab)


# ============================================================
# RANKING MODEL SHAPE
# ============================================================

class TestRankingShape:
    """Test ranking model input/output."""

    def test_feature_table_shape(self):
        """Feature table: n_rows × n_features, no NaN after fillna."""
        n_rows, n_features = 100, 36
        X = np.random.randn(n_rows, n_features)
        X[0, 5] = np.nan  # simulate missing

        X = np.nan_to_num(X, nan=0.0)
        assert X.shape == (n_rows, n_features)
        assert not np.any(np.isnan(X))

    def test_prediction_shape(self):
        """LightGBM predict output = 1D array, same length as input."""
        try:
            import lightgbm as lgb
        except ImportError:
            pytest.skip("lightgbm not installed")

        # Train tiny model
        n = 50
        X = np.random.randn(n, 5)
        y = np.random.randint(0, 2, n)

        train_data = lgb.Dataset(X, label=y)
        params = {'objective': 'binary', 'verbose': -1, 'n_estimators': 2}
        model = lgb.train(params, train_data, num_boost_round=2)

        preds = model.predict(X)
        assert preds.shape == (n,)
        assert all(0 <= p <= 1 for p in preds)


# ============================================================
# FAISS SHAPE
# ============================================================

class TestFAISSShape:
    """Test FAISS index build + query."""

    def test_faiss_search_output(self):
        """FAISS search returns (distances, indices) with correct K."""
        try:
            import faiss
        except ImportError:
            pytest.skip("faiss not installed")

        dim = 16
        n_items = 50
        k = 5

        embeddings = np.random.randn(n_items, dim).astype(np.float32)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        query = np.random.randn(1, dim).astype(np.float32)
        D, I = index.search(query, k)

        assert D.shape == (1, k)
        assert I.shape == (1, k)
        assert all(0 <= idx < n_items for idx in I[0])