import requests

url = 'https://api.fpt.ai/hmi/asr/general'
payload = open('voices/test.mp3', 'rb').read()
headers = {
    'api-key': 'lNbxag5GVRA3ZYBYzjeijcmY2BSAChdr'
}

response = requests.post(url=url, data=payload, headers=headers)

print(response.json())
