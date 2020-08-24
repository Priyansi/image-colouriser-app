import requests

resp = requests.post("http://localhost:5000/predict",
                     files={'file': open('test.jpg', 'rb')})

print(resp.text)
