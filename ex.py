import requests
host = "http://135.181.63.141:7777/get_spans"
print(requests.post(host, json={"text": "This is outrageous. This is unfair and ugly"}).json())

"""
Response json
{'sentences': ['[[This is outrageous]].', 'This is [[unfair and ugly]]'], 'spans': [[0, 18], [27, 42]]}
"""