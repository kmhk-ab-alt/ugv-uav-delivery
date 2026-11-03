"""Wire format validation, independent of ROS."""
import json

def decode_command(payload, expected_round):
    data = json.loads(payload)
    if type(data.get('round')) is not int or data['round'] != expected_round:
        raise ValueError('stale or invalid round')
    if data.get('action') not in ('smash', 'dodge'):
        raise ValueError('invalid action')
    return data['action'], str(data.get('reason', ''))[:300]
