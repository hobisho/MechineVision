import base64
import requests

# OpenAI API Key
api_key = ""

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "l.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o-mini", #可更改為以下模型gpt-4o、gpt-4o-mini、o4-mini、o1
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "請問照片中的內容是甚麼?請用台灣標準用語繁體中文來呈現詳細說明。"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
            "detail": "high"
          }
        }
      ]
    }
  ]
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
a=response.json()

print(a['choices'][0]['message']['content'])