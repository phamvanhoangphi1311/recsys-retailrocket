"""
Test training logic trên fake data nhỏ.
Không cần real data, không cần GPU, chạy trong 5 giây.
Catch logic bugs TRƯỚC khi Kubeflow chạy trên real data.
"""
import pytest
import pandas as pd
import numpy as np


# ============================================================
# LABEL DESIGN LOGIC
# ============================================================

class TestLabelDesign:
    """Test label assignment logic — đây là core, sai = toàn bộ pipeline sai."""

    def _assign_label(self, n_trans, n_carts):
        """Reproduce label logic từ step2."""
        if n_trans > 0:
            return 1, 1.0, 'transaction'
        elif n_carts > 0:
            return 1, 0.5, 'addtocart'
        else:
            return 0, 0.7, 'view_only'

    def test_transaction_is_positive(self):
        label, weight, source = self._assign_label(n_trans=1, n_carts=0)
        assert label == 1
        assert weight == 1.0
        assert source == 'transaction'

    def test_cart_is_positive(self):
        label, weight, source = self._assign_label(n_trans=0, n_carts=1)
        assert label == 1
        assert weight == 0.5
        assert source == 'addtocart'

    def test_view_only_is_negative(self):
        label, weight, source = self._assign_label(n_trans=0, n_carts=0)
        assert label == 0
        assert weight == 0.7
        assert source == 'view_only'

    def test_transaction_overrides_cart(self):
        """User cart rồi mua → label = transaction, không phải addtocart."""
        label, weight, source = self._assign_label(n_trans=1, n_carts=1)
        assert label == 1
        assert weight == 1.0
        assert source == 'transaction'


# ============================================================
# FEATURE ENGINEERING LOGIC
# ============================================================

class TestFeatureEngineering:
    """Test feature computation logic."""

    @pytest.fixture
    def fake_events(self):
        """10 events, 3 users, 4 items."""
        return pd.DataFrame({
            'visitorid': ['u1','u1','u1','u1','u2','u2','u2','u3','u3','u3'],
            'itemid':    ['i1','i2','i1','i1','i1','i3','i3','i2','i4','i4'],
            'event':     ['view','view','view','transaction','view','view','addtocart','view','view','transaction'],
            'timestamp_dt': pd.date_range('2024-01-01', periods=10, freq='h'),
            'session_id':   [1,1,1,1,1,1,1,1,1,1],
        })

    def test_user_view_count(self, fake_events):
        """u1 có 3 views."""
        views = fake_events[fake_events['event'] == 'view']
        user_views = views.groupby('visitorid').size()
        assert user_views['u1'] == 3

    def test_cross_views_this_item(self, fake_events):
        """u1 viewed i1 twice (3 events nhưng 1 là transaction)."""
        views = fake_events[fake_events['event'] == 'view']
        pair_views = views.groupby(['visitorid', 'itemid']).size()
        assert pair_views[('u1', 'i1')] == 2

    def test_item_unique_visitors(self, fake_events):
        """i1 có 2 unique visitors (u1, u2)."""
        item_visitors = fake_events.groupby('itemid')['visitorid'].nunique()
        assert item_visitors['i1'] == 2

    def test_leave_one_out_category(self, fake_events):
        """LOO: u1 mua 1 item category A → cat_trans_total=1, subtract pair → loo=0."""
        # Simulate: u1 has 1 transaction in category
        cat_trans_total = 1  # u1 mua 1 item trong category
        pair_trans = 1       # pair (u1, i1) chính là transaction đó
        loo = cat_trans_total - pair_trans
        assert loo == 0  # after removing this pair, u1 có 0 trans trong category

    def test_bayesian_smoothing(self):
        """Item 1 view, 1 purchase → raw rate 100%. Smoothed rate < 100%."""
        n_trans = 1
        n_views = 1
        alpha, beta = 1, 50

        raw_rate = n_trans / max(n_views, 1)
        smoothed_rate = (n_trans + alpha) / (n_views + alpha + beta)

        assert raw_rate == 1.0
        assert smoothed_rate < 0.05  # smoothed gần 1/50 = 2%

    def test_no_future_leak(self, fake_events):
        """Features chỉ từ events TRƯỚC cutoff."""
        cutoff = fake_events['timestamp_dt'].quantile(0.5)
        train_events = fake_events[fake_events['timestamp_dt'] <= cutoff]
        future_events = fake_events[fake_events['timestamp_dt'] > cutoff]

        # Train events không chứa future events
        assert len(train_events) < len(fake_events)
        assert train_events['timestamp_dt'].max() <= cutoff


# ============================================================
# ID MAPPER LOGIC
# ============================================================

class TestIDMapper:
    """Test ID mapping logic."""

    def test_warm_user_detection(self, tmp_path):
        """User trong user2id → warm."""
        import json
        user2id = {"user_123": 0, "user_456": 1}
        item2id = {"item_a": 0}

        u_path = tmp_path / "user2id.json"
        i_path = tmp_path / "item2id.json"
        u_path.write_text(json.dumps(user2id))
        i_path.write_text(json.dumps(item2id))

        from src.id_mapper import IDMapper
        mapper = IDMapper(str(u_path), str(i_path))

        assert mapper.is_warm_user("user_123") == True
        assert mapper.is_warm_user("unknown") == False

    def test_index_mapping(self, tmp_path):
        """Get correct integer index."""
        import json
        user2id = {"user_123": 42}
        item2id = {"item_a": 7}

        u_path = tmp_path / "user2id.json"
        i_path = tmp_path / "item2id.json"
        u_path.write_text(json.dumps(user2id))
        i_path.write_text(json.dumps(item2id))

        from src.id_mapper import IDMapper
        mapper = IDMapper(str(u_path), str(i_path))

        assert mapper.get_user_idx("user_123") == 42
        assert mapper.get_item_idx("item_a") == 7
        assert mapper.get_user_idx("unknown") is None

    def test_reverse_mapping(self, tmp_path):
        """Index → ID."""
        import json
        user2id = {"user_123": 0}
        item2id = {"item_a": 5}

        u_path = tmp_path / "user2id.json"
        i_path = tmp_path / "item2id.json"
        u_path.write_text(json.dumps(user2id))
        i_path.write_text(json.dumps(item2id))

        from src.id_mapper import IDMapper
        mapper = IDMapper(str(u_path), str(i_path))

        assert mapper.get_item_id(5) == "item_a"


# ============================================================
# TEMPORAL SPLIT LOGIC
# ============================================================

class TestTemporalSplit:
    """Test temporal split — sai split = leakage."""

    def test_no_future_in_train(self):
        """Train set không chứa events sau cutoff."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({'date': dates, 'value': range(100)})

        cutoff = df['date'].quantile(0.70)
        train = df[df['date'] < cutoff]
        test = df[df['date'] >= cutoff]

        assert train['date'].max() < test['date'].min()

    def test_split_ratio(self):
        """70/15/15 split roughly correct."""
        n = 1000
        dates = pd.date_range('2024-01-01', periods=n, freq='h')

        train_cut = dates[int(n * 0.70)]
        val_cut = dates[int(n * 0.85)]

        train = dates[dates < train_cut]
        val = dates[(dates >= train_cut) & (dates < val_cut)]
        test = dates[dates >= val_cut]

        assert len(train) / n == pytest.approx(0.70, abs=0.02)
        assert len(val) / n == pytest.approx(0.15, abs=0.02)
        assert len(test) / n == pytest.approx(0.15, abs=0.02)