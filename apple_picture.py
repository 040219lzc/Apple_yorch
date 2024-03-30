# 刘子淳

# 生来就比别的孩子来的犟，长大就要挑比别人大的台子上。

import requests
import json
from bs4 import BeautifulSoup
import time

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0',
    'Cookie':'Hm_lvt_d60c24a3d320c44bcd724270bc61f703=1710256448; Hm_lpvt_d60c24a3d320c44bcd724270bc61f703=1710258971',
    'Referer':'https://soso.nipic.com/?q=%E7%BA%A2%E8%8B%B9%E6%9E%9C',
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
}

url = "https://cn.bing.com/images/search?q=%E7%BA%A2%E8%8B%B9%E6%9E%9C%E5%9B%BE%E7%89%87&form=HDRSC2&first=1&cw=1225&ch=1323"
url2 = "https://soso.nipic.com/?q=%E7%BA%A2%E8%8B%B9%E6%9E%9C&g=0&or=0&f=JPG&y=60"
resp = requests.get(url)
resp2 = requests.get(url2,headers=headers)
resp.encoding = 'utf-8'
resp2.encoding = 'utf-8'

#print(resp.text)

main_page = BeautifulSoup(resp.text, "html.parser")
main_page2 = BeautifulSoup(resp2.text, "html.parser")
alist = main_page.find("div", class_="dgControl dtl hover").find_all("img")
alist2 = main_page2.find("div", class_="search-works-wrap").find_all("img")

list = []
list_end = []

#print(alist)
#print(alist2)

for a in alist:
    if a.get('data-original') is not None:
        img1 = a.get('src')
        #img1.content
        img_resp1 = requests.get(img1)
        img_name1 = img1.split("/")[-1]
        with open("img/"+img_name1,mode="wb") as f:
            f.write(img_resp1.content)

        print("over!!!",img_name1)
        time.sleep(1)


for b in alist2:
    if b.get('data-original') is not None:
        #new_url = b.get('data-original')
        new_url = b.get('data-original')
        img2 = "http:" + new_url
        img_resp2 = requests.get(img2)
        img_resp2.content
        img_name2 = img2.split("/")[-1]
        with open("img/"+img_name2, mode="wb") as g:
            g.write(img_resp2.content)

        print("over!!!", img_name2)
        time.sleep(1)

print("all over")


