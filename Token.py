import uuid

# An easy way to save the nodes
class Token():
    def __init__(self, token_id, value, prev_token=None, next_tokens=None, metric='entropy', metric_value=None, is_forking_token=False, part_of_response=True):
        self.id = str(uuid.uuid4())
        self.token_id = token_id
        self.value = value
        self.prev_token = prev_token
        self.next_tokens = next_tokens if next_tokens else []
        self.metric = metric
        self.metric_value = metric_value
        self.is_forking_token = is_forking_token
        self.part_of_response = part_of_response

    def add_prev_token(self, prev_token):
        self.prev_token = prev_token
    
    def add_next_token(self, next_token):
        self.next_tokens.append(next_token)