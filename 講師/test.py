import openai

def main():
    # 設置你的 OpenAI API 密鑰
    response_list="這是我們的"
    openai.api_key = ''
    i=0
    while True:
        i=i+1
        user_input = input("請輸入對話內容（輸入 '結束' 以結束程式）: ")
        response_list = response_list + f"{i} times question: {user_input}\n"
        # 從使用者獲取輸入
        
        if user_input.lower() == '結束':
            break

        # 檢查是否包含 '產生文宣' 關鍵句
        # 移除 '產生文宣' 及其之後的內容
        # user_input = user_input.split('')[0]

        # 使用 GPT 模型進行對話
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # 可更改為以下模型gpt-4o、gpt-4o-mini、o3-mini、o4-mini、o1
            messages=[
                {"role": "user", "content": response_list}
            ]
        )
        # 輸出回覆的內容
        response_list = response_list + f"{i} times question: {response['choices'][0]['message']['content']}\n"
        print(response_list)
        print(response['choices'][0]['message']['content'])

if __name__ == "__main__":
    main()