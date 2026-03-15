"""
ID Mapper — convert between visitorid/itemid strings and integer indices.
Used by serving code to map user_id from request to embedding index.
"""

import json

class IDMapper:
    def __init__(self, user2id_path, item2id_path):
        with open(user2id_path) as f:
            self.user2id = json.load(f)
        with open(item2id_path) as f:
            self.item2id = json.load(f)
        self.id2user = {v: k for k, v in self.user2id.items()}
        self.id2item = {v: k for k, v in self.item2id.items()}
    
    def is_warm_user(self, user_id):
        return user_id in self.user2id
    
    def get_user_idx(self, user_id):
        return self.user2id.get(user_id)
    
    def get_item_idx(self, item_id):
        return self.item2id.get(item_id)
    
    def get_item_id(self, idx):
        return self.id2item.get(idx)
