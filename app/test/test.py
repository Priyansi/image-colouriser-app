import requests

resp = requests.post("https://image-colouriser.herokuapp.com/predict",
                     files={'file': open('test.jpg', 'rb')})

print(resp.text)
