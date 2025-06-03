# -*- coding: utf-8 -*-
import requests

# 設定 OpenAI API 金鑰
api_key = ''

def text_to_speech(text):
    # 設定API的End-point
    url = 'https://api.openai.com/v1/audio/speech'

    # HTTP的Header 
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # TTS相關模型參數設定
    data = {
        'model': 'gpt-4o-mini-tts',
        'input': text,
        'voice': 'echo',  # 人聲可以選擇：'nova', 'shimmer', 'echo', 'onyx', 'fable', 'alloy', 'ash', 'sage', 'coral', 'ballad'
        'stream': True,  # 如需要產生聲音串流，則取消註解。
    }

    # 發送API服務請求
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        # 處理回應，如存檔或直接撥放串流
        audio = response.content
        # 存檔
        with open('output.mp3', 'wb') as file:
            file.write(audio)
        return "語音檔案儲存完成。"
    else:
        return "錯誤: " + response.text

# 使用示例
text = """我個人認為
義大利麵就應該拌 42 號混泥土
因為這個螺絲釘的長度很容易直接影響到挖掘機的扭矩

你往裡砸的時候
一瞬間他就會產生大量的高能蛋白
俗稱 UFO，會嚴重影響經濟的發展
以至於對整個太平洋和充電器的核污染

再或者說
透過這勾股定理
很容易推斷出人工飼養的東條英機
他是可以捕獲野生的三角函數

所以說
不管這秦始皇的切面是否具有放射性
川普的 N 次方是否有沈澱物
都不會影響到沃爾瑪跟維爾康在南極匯合

你往裡砸的時候
一瞬間他就會產生大量的高能蛋白
俗稱 UFO，會嚴重影響經濟的發展
以至於對整個太平洋和充電器的核污染

再或者說
透過這勾股定理
很容易推斷出人工飼養的東條英機
他是可以捕獲野生的三角函數

所以說
不管這秦始皇的切面是否具有放射性
川普的 N 次方是否有沈澱物
都不會影響到沃爾瑪跟維爾康在南極匯合"""
result = text_to_speech(text)
print(result)