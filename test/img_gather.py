import requests
from bs4 import BeautifulSoup
import random

def image(word,i):
    response = requests.get("https://www.google.co.jp/search?q="+word+"&tbm=isch") #HTMLの取得
    soup = BeautifulSoup(response.text,"lxml") #HTMLの解析？変換？
    links = soup.find_all("img") #img タグですべて抜き出し
    link = links[i].get("src")
    return link

def download_img(url, file_name):
    r = requests.get(url, stream=True) #stream=Trueは処理を速める
    if r.status_code == 200:# webpageが機能してるか判断 例えば存在しない場合は404となる
        with open(file_name+".jpeg", 'wb') as f:#open(ファイル名,何をするか) as 一時ファイル名（fが一般的）
            f.write(r.content)

def code():
    code = ""
    for i in range(10):
        code += random.choice("0123456789")
    return code

num = input("検索回数:")
word = input("検索ワード:")
for i in range(1,int(num)+1): #0から始めるとグーグルのアイコンを読み込んでバグる（謎）
    link = image(word,i)
    download_img(link,str(i))
print("OK")
