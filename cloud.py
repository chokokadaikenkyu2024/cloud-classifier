from flask import Flask, request, jsonify, send_file, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage, ImageSendMessage, MessageEvent, TextMessage, ImageMessage
from linebot.exceptions import InvalidSignatureError
import numpy as np
import os
import cv2
import requests
from datetime import datetime, timezone, timedelta
import pytz
from tensorflow.keras.models import load_model

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = 'zizBFLqOl3a/SO/KW9pZklc16wc9T0eI/40ZoC8zka4MdUCqJhTbaoQkiRqaNiH1ylpt4TyXngLBLiW3VPH/U4kEfHTpuQ3a5YWkGzYtholnJHh3AGoJIYnd3ST52xIZQ0MfKu700u3gj6XwtaGXAwdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = 'b7bb2fb431b310e7cdae51dbe5f92293'
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

LATITUDE = 36.6513
LONGITUDE = 138.1810
last_update_date = None
sunrise_jst = None
sunset_jst = None
kumolog = None
kumolog2 = None
rain_cnt=0
#実行関数
classes = ["Ac", "As", "Bs", "Cb", "Cc", "Ci", "Cs", "Cu", "Ns", "Sc", "St"]
image_size = 100
model_path = '/home/chokokadaikenkyu/mysite/cloud.h5'
static_folder = '/home/chokokadaikenkyu/mysite/static'
temp_model_path = '/home/chokokadaikenkyu/mysite/temp.h5'
save_path = os.path.join(static_folder, 'photo.jpg')

image_url = 'https://chokokadaikenkyu.pythonanywhere.com/static/photo.jpg'

prediction_file = os.path.join(static_folder, 'prediction.txt')
#通知関数
raincnt=0
prehour = 99
temp=None

#日の出日没時間取得
def get_sunrise_sunset(lat, lon):
    url = "https://api.sunrise-sunset.org/json"
    params = {
        "lat": lat,
        "lng": lon,
        "formatted": 0
    }
    response = requests.get(url, params=params)
    data = response.json()
    if response.status_code == 200 and data["status"] == "OK":
        return data["results"]["sunrise"], data["results"]["sunset"]#日の出日没時間取得関数
    else:
        raise Exception("APIリクエストに失敗しました。")

def convert_to_jst_datetime(utc_time):
    utc_dt = datetime.fromisoformat(utc_time.replace("Z", "+00:00"))
    jst_dt = utc_dt.astimezone(timezone(timedelta(hours=9)))
    return jst_dt

def update_sunrise_sunset():
    global last_update_date, sunrise_jst, sunset_jst
    now_date = datetime.now(timezone(timedelta(hours=9))).date()
    if last_update_date != now_date:
        sunrise, sunset = get_sunrise_sunset(LATITUDE, LONGITUDE)
        sunrise_jst = convert_to_jst_datetime(sunrise)
        sunset_jst = convert_to_jst_datetime(sunset)
        last_update_date = now_date
        print("日の出・日没時間を再取得しました。")

@app.route('/')
def index():
    return send_file('/home/chokokadaikenkyu/mysite/templates/index.html')

@app.route('/save_image', methods=['POST'])
def save_image():
    global kumolog
    global kumolog2
    global rain_cnt
    global prehour
    global temp
    now = datetime.now(pytz.timezone('Asia/Tokyo'))
    current_hour = now.hour
    current_minutes = now.minute
    image_url = 'https://chokokadaikenkyu.pythonanywhere.com/static/photo.jpg'

    if 'image' not in request.files:
        return jsonify({'error': '画像ファイルがありません'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '選択されたファイルがありません'}), 400

    try:
        file.save(save_path)
        img = cv2.imread(save_path)
        height, width = img.shape[:2]
        trimmed_image = img[:int(height * 0.9), :]  # 上部90%をトリム
        resized_image = cv2.resize(trimmed_image, (image_size, image_size))  # リサイズ
        if current_hour != prehour:
            model = load_model(temp_model_path)
            month = now.month
            day = now.day
            hour_of_day = now.hour
            input_data = np.array([[hour_of_day, day, month]])
            predicted_temp = model.predict(input_data)
            temp = predicted_temp[0][0]
            prehour = current_hour

        # グレースケールとカラー画像の作成
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # グレースケール化
        gray_image_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # グレースケールをRGBに変換
        data_combined = (gray_image_rgb.astype(np.float32) / 255.0 + resized_image.astype(np.float32) / 255.0) / 2

        model = load_model(model_path)
        predictions = model.predict(np.expand_dims(data_combined, axis=0))
        predicted_class_name = classes[np.argmax(predictions, axis=-1)[0]]

        # 予測結果をファイルに保存
        with open(prediction_file, 'w') as f:
            f.write(predicted_class_name)



        # 雨予想メッセージの送信
        if 7 <= current_hour < 18 and predicted_class_name in ["Cb", "Ns"]:
            rain_cnt+=1
            if temp>3 and kumolog2==kumolog==predicted_class_name and rain_cnt==3:
                text = TextSendMessage(text=f"☔雨予想☔\n{'積乱雲' if predicted_class_name == 'Cb' else '乱層雲'}が近くにあります。\n雨にご注意ください")
                image = ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
                line_bot_api.broadcast(messages=[text, image])

            elif kumolog2==kumolog==predicted_class_name and rain_cnt==3:
                text = TextSendMessage(text=f"⛄雪予想⛄{'積乱雲' if predicted_class_name == 'Cb' else '乱層雲'}が近くにあります。\n雪にご注意ください。")
                image = ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
                line_bot_api.broadcast(messages=[text, image])
        else:
            rain_cnt=0
        kumolog2=kumolog
        kumolog=predicted_class_name





        if temp>3:
            if current_hour == 11 and current_minutes == 10:
                morning_texts = {
                    "Ac": "現在、高積雲と識別されています。これから雨が降るかもしれません。",
                    "As": "現在、高層雲と識別されています。これから雨が降る可能性があります。",
                    "Bs": "現在、雲がない快晴と識別されています。快晴が続くでしょう。",
                    "Cb": "現在、積乱雲と識別されています。これから雨が降る可能性が高いです。",
                    "Cc": "現在、巻積雲と識別されています。これから雨が降るかもしれません。",
                    "Ci": "現在、巻雲と識別されています。これから雨が降るかもしれません。",
                    "Cs": "現在、巻層雲と識別されています。これから雨が降るかもしれません。",
                    "Cu": "現在、積雲と識別されています。晴れが続くでしょう。",
                    "Ns": "現在、乱層雲と識別されています。これから雨が降る可能性が高いです。",
                    "Sc": "現在、層積雲と識別されています。曇りが続くでしょう。",
                    "St": "現在、層雲と識別されています。これから雨が降る可能性があります。"
                }
                if predicted_class_name in morning_texts:
                    text = TextSendMessage(text=f"おはようございます！\n{morning_texts[predicted_class_name]}")
                    image = ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
                    line_bot_api.broadcast(messages=[text, image])

            if current_hour == 13 and current_minutes == 20:
                afternoon_texts = {
                    "Ac": "現在、高積雲と識別されています。これから雨が降るかもしれません。",
                    "As": "現在、高層雲と識別されています。これから雨が降る可能性があります。",
                    "Bs": "現在、雲がない快晴と識別されています。快晴が続くでしょう。",
                    "Cb": "現在、積乱雲と識別されています。これから雨が降る可能性が高いです。",
                    "Cc": "現在、巻積雲と識別されています。これから雨が降るかもしれません。",
                    "Ci": "現在、巻雲と識別されています。これから雨が降るかもしれません。",
                    "Cs": "現在、巻層雲と識別されています。これから雨が降るかもしれません。",
                    "Cu": "現在、積雲と識別されています。晴れが続くでしょう。",
                    "Ns": "現在、乱層雲と識別されています。これから雨が降る可能性が高いです。",
                    "Sc": "現在、層積雲と識別されています。曇りが続くでしょう。",
                    "St": "現在、層雲と識別されています。これから雨が降る可能性があります。"
                }
                if predicted_class_name in afternoon_texts:
                    text = TextSendMessage(text=f"こんにちは！\n{afternoon_texts[predicted_class_name]}")
                    image = ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
                    line_bot_api.broadcast(messages=[text, image])
        else:
            if current_hour == 11 and current_minutes == 10:
                morning_texts = {
                    "Ac": "現在、高積雲と識別されています。これから雪が降るかもしれません。",
                    "As": "現在、高層雲と識別されています。これから雪が降る可能性があります。",
                    "Bs": "現在、雲がない快晴と識別されています。快晴が続くでしょう。",
                    "Cb": "現在、積乱雲と識別されています。これから雪が降る可能性が高いです。",
                    "Cc": "現在、巻積雲と識別されています。これから雪が降るかもしれません。",
                    "Ci": "現在、巻雲と識別されています。これから雪が降るかもしれません。",
                    "Cs": "現在、巻層雲と識別されています。これから雪が降るかもしれません。",
                    "Cu": "現在、積雲と識別されています。晴れが続くでしょう。",
                    "Ns": "現在、乱層雲と識別されています。これから雪が降る可能性が高いです。",
                    "Sc": "現在、層積雲と識別されています。曇りが続くでしょう。",
                    "St": "現在、層雲と識別されています。これから雪が降る可能性があります。"
                }
                if predicted_class_name in morning_texts:
                    text = TextSendMessage(text=f"おはようございます！\n{morning_texts[predicted_class_name]}")
                    image = ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
                    line_bot_api.broadcast(messages=[text, image])

            if current_hour == 14 and current_minutes == 45:
                afternoon_texts = {
                    "Ac": "現在、高積雲と識別されています。これから雪が降るかもしれません。",
                    "As": "現在、高層雲と識別されています。これから雪が降る可能性があります。",
                    "Bs": "現在、雲がない快晴と識別されています。快晴が続くでしょう。",
                    "Cb": "現在、積乱雲と識別されています。これから雪が降る可能性が高いです。",
                    "Cc": "現在、巻積雲と識別されています。これから雪が降るかもしれません。",
                    "Ci": "現在、巻雲と識別されています。これから雪が降るかもしれません。",
                    "Cs": "現在、巻層雲と識別されています。これから雪が降るかもしれません。",
                    "Cu": "現在、積雲と識別されています。晴れが続くでしょう。",
                    "Ns": "現在、乱層雲と識別されています。これから雪が降る可能性が高いです。",
                    "Sc": "現在、層積雲と識別されています。曇りが続くでしょう。",
                    "St": "現在、層雲と識別されています。これから雪が降る可能性があります。"
                }
                if predicted_class_name in afternoon_texts:
                    text = TextSendMessage(text=f"こんにちは！\n{afternoon_texts[predicted_class_name]}")
                    image = ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
                    line_bot_api.broadcast(messages=[text, image])
        return jsonify({'message': '画像が保存され、識別が行われました。'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

#テキストメッセージ　　
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    update_sunrise_sunset()
    now = datetime.now(pytz.timezone('Asia/Tokyo'))
    text = event.message.text
    image_url = 'https://chokokadaikenkyu.pythonanywhere.com/static/photo.jpg'
    model = load_model(temp_model_path)
    month = now.month
    day = now.day
    hour_of_day = now.hour
    if now.minute >= 30:
        hour_of_day += 1
    input_data = np.array([[hour_of_day, day, month]])
    predicted_temp = model.predict(input_data)
    temp =predicted_temp[0][0]

    if text == "今の空は":
        if sunrise_jst <= now <= sunset_jst:
            try:
                    line_bot_api.reply_message(
                        event.reply_token,
                        ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
                    )
            except Exception as e:
                print(f"Error: {e}")
                line_bot_api.reply_message(
                     event.reply_token,
                    TextSendMessage(text="画像の取得中にエラーが発生しました。")
                )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=f"{sunrise_jst.hour}:{sunrise_jst.minute}～{sunset_jst.hour}:{sunset_jst.minute}は休止しています。")
            )
    elif text == "今の天気は":
        if sunrise_jst <= now <= sunset_jst:
            try:
                if os.path.exists(prediction_file):
                    with open(prediction_file, 'r') as f:
                        predicted_class_name = f.read().strip()
                    if temp>3:
                        result_text = {
                            "Ac": "現在、高積雲と識別されています。これから雨が降るかもしれません。",
                            "As": "現在、高層雲と識別されています。これから雨が降る可能性があります。",
                            "Bs": "現在、雲がない快晴と識別されています。快晴が続くでしょう。",
                            "Cb": "現在、積乱雲と識別されています。これから雨が降る可能性が高いです。",
                            "Cc": "現在、巻積雲と識別されています。これから雨が降るかもしれません。",
                            "Ci": "現在、巻雲と識別されています。これから雨が降るかもしれません。",
                            "Cs": "現在、巻層雲と識別されています。これから雨が降るかもしれません。",
                            "Cu": "現在、積雲と識別されています。晴れが続くでしょう。",
                            "Ns": "現在、乱層雲と識別されています。これから雨が降る可能性が高いです。",
                            "Sc": "現在、層積雲と識別されています。曇りが続くでしょう。",
                            "St": "現在、層雲と識別されています。これから雨が降る可能性があります。"
                        }.get(predicted_class_name, "未知のクラスです。")
                    else:
                        result_text = {
                            "Ac": "現在、高積雲と識別されています。これから雪が降るかもしれません。",
                            "As": "現在、高層雲と識別されています。これから雪が降る可能性があります。",
                            "Bs": "現在、雲がない快晴と識別されています。快晴が続くでしょう。",
                            "Cb": "現在、積乱雲と識別されています。これから雪が降る可能性が高いです。",
                            "Cc": "現在、巻積雲と識別されています。これから雪が降るかもしれません。",
                            "Ci": "現在、巻雲と識別されています。これから雪が降るかもしれません。",
                            "Cs": "現在、巻層雲と識別されています。これから雪が降るかもしれません。",
                            "Cu": "現在、積雲と識別されています。晴れが続くでしょう。",
                            "Ns": "現在、乱層雲と識別されています。これから雪が降る可能性が高いです。",
                            "Sc": "現在、層積雲と識別されています。曇りが続くでしょう。",
                            "St": "現在、層雲と識別されています。これから雪が降る可能性があります。"
                        }.get(predicted_class_name, "未知のクラスです。")

                    line_bot_api.reply_message(
                        event.reply_token,
                        [
                            TextSendMessage(text=result_text),
                            ImageSendMessage(original_content_url=image_url, preview_image_url=image_url),
                        ]
                    )
                else:
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text="識別結果がまだありません。")
                    )
            except Exception as e:
                print(f"Error: {e}")
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="天気の取得中にエラーが発生しました。")
                )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=f"{sunrise_jst.hour}:{sunrise_jst.minute}～{sunset_jst.hour}:{sunset_jst.minute}は休止しています。")
            )
    elif text == "気温":
        now = datetime.now(pytz.timezone('Asia/Tokyo'))
        model = load_model(temp_model_path)
        month = now.month
        day = now.day
        hour_of_day = now.hour
        if now.minute >= 30:
            hour_of_day += 1
        input_data = np.array([[hour_of_day, day, month]])
        predicted_temp = model.predict(input_data)
        tempt = round(predicted_temp[0][0], 2)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f" {tempt:.2f}℃予測されました。")
        )

    elif text=="時間":
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"長野市の日の出: {sunrise_jst.hour}:{sunrise_jst.minute}\n長野市の日没: {sunset_jst.hour}:{sunset_jst.minute}\n現在の時刻: {now.hour}:{now.minute}")
        )
    elif text=="雨":
        global rain_cnt
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"{rain_cnt}")
        )
    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="対応している文章ではありません。\n「今の空は」と送信すると、最新の定点カメラの画像を見ることができます。\n「今の天気は」と送信すると、最新の定点カメラの識別結果と画像、現在の気温予測を見ることができます。\n「気温」と送信すると、現在の気温予測を見ることができます。\n「時間」と送信すると、今日の長野市の日の出時刻、日没時刻、現在時刻を見ることができます。\nまた、画像を送信するとその画像の雲の種類を見ることができます。")
        )

#画像受信
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    try:
        message_content = line_bot_api.get_message_content(event.message.id)
        image_data = np.frombuffer(message_content.content, np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        trimmed_gray = gray_image[:int(gray_image.shape[0] * 0.9), :]
        resized_gray = cv2.resize(trimmed_gray, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
        gray_image_rgb = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)
        trimmed_color = image[:int(image.shape[0] * 0.9), :]
        resized_color = cv2.resize(trimmed_color, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
        data_gray_rgb = gray_image_rgb.astype(np.float32) / 255.0
        data_color = resized_color.astype(np.float32) / 255.0
        data_combined = (data_gray_rgb + data_color) / 2.0
        X = np.array([data_combined])
        model = load_model(model_path)
        predictions = model.predict(X)
        predicted_class_name = classes[np.argmax(predictions, axis=-1)[0]]
        result_text = {
            "Ac": "高積雲と予測されました。これから雨が降るかもしれません。",
            "As": "高層雲と予測されました。これから雨が降る可能性があります。",
            "Bs": "雲がない快晴と予測されました。快晴が続くでしょう。",
            "Cb": "積乱雲と予測されました。これから雨が降る可能性が高いです。",
            "Cc": "巻積雲と予測されました。これから雨が降るかもしれません。",
            "Ci": "巻雲と予測されました。これから雨が降るかもしれません。",
            "Cs": "巻層雲と予測されました。これから雨が降るかもしれません。",
            "Cu": "積雲と予測されました。晴れが続くでしょう。",
            "Ns": "乱層雲と予測されました。これから雨が降る可能性が高いです。",
            "Sc": "層積雲と予測されました。曇りが続くでしょう。",
            "St": "層雲と予測されました。これから雨が降る可能性があります。"
        }
        result_message = result_text.get(predicted_class_name, "未知のクラスです。")
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result_message)
        )
    except Exception as e:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"画像処理中にエラーが発生しました: {str(e)}")
        )
#実行

if __name__ == "__main__":
    app.run()