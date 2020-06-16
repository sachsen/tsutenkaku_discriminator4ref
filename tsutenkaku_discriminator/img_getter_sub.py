import requests
from bs4 import BeautifulSoup
import random

def image(word,i):#インターネット上から画像データを取得する関数
    response = requests.get("https://www.google.co.jp/search?q="+word+"&tbm=isch") #HTMLの取得
    soup = BeautifulSoup(response.text,"lxml") #HTMLの解析？変換？
    links = soup.find_all("img") #img タグですべて抜き出し
    link = links[i].get("src")
    return link
def download_img(url, file_name, path):#画像をダウンロードする関数
    r = requests.get(url, stream=True) #stream=Trueは処理を速める
    if r.status_code == 200:# webpageが機能してるか判断 例えば存在しない場合は404となる
        with open(path+file_name + ".jpeg", 'wb') as f:#open(ファイル名,何をするか) as 一時ファイル名（fが一般的）
            f.write(r.content)
def distribute_img(num,word,path1,path2,ratio):#画像を分配する関数
    random_list = random.choices(["a","b"],weights = (ratio),k=int(num))
    for i in range(1,int(num)+1):  #0から始めるとグーグルのアイコンを読み込んでバグる（謎）
        link = image(word,i)
        if random_list[i-1] == "a":
            download_img(link,str(i),path1)
        if random_list[i-1] == "b":
            download_img(link,str(i),path2)

##############ここから下をいじっってくれればよいです###############################################################################

num = 1000   #ここに画像枚数 int型
word = "太陽の塔"  #ここに画像の検索ワード str型
path_1 = "./data/taiyo_no_to/"  #ここに一つ目の保存場所を入力　注意点として\を/に置換すること また/で終わること　例(desktop):C:/Users/ユーザー名/Desktop/
path_2 = "./data/taiyo_no_to_valid/"  #ここに二つ目の保存場所を入力　注意点として\を/に置換すること また/で終わること　例(desktop):C:/Users/ユーザー名/Desktop/
ratio = 9,1           #ここに保存場所１:保存場所２ の比を書く　「:」の代わりに「,」を使うこと 要するにlist型で書くこと

distribute_img(num,word,path_1,path_2,ratio) #この関数一つで機能するようにした　引数は(画像数、検索ワード、保存場所1、保存場所2、比)

print("OK")
