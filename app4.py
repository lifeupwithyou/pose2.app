from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import mediapipe as mp
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'  # 保存先ディレクトリ
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # ディレクトリが存在しない場合は作成

# MediaPipeのPoseモジュールの初期化
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# 開発者が用意しないといけないファイル
# 一度だけロードにしてコスト削減
model = load_model("pose_model.keras")  

@app.route('/')
def index():
    return render_template('index.html', title="Pose Prediction", message="写真をアップロードしてください。")


@app.route('/index', methods=['POST'])
def predict():
    file = request.files['file']  # ユーザーがuploadした画像
    uploaded_image_url = None  # 初期化
  
    if file:
         # 画像を保存
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # 画像のURLを取得
        uploaded_image_url = url_for('static', filename=f'uploads/{file.filename}')
        
         # 画像をMediaPipe形式に変換して読み取る
        image = Image.open(file)
        image_rgb = np.array(image.convert("RGB")) #画像をRGB形式
        results = pose.process(image_rgb) #画像内の体の部位の座標（ランドマーク）を検出

         # 特定の座標のみ抽出
        pick_up = {
            "LEFT_SHOULDER": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "RIGHT_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "LEFT_ELBOW": mp_pose.PoseLandmark.LEFT_ELBOW,
            "RIGHT_ELBOW": mp_pose.PoseLandmark.RIGHT_ELBOW,
            "LEFT_HIP": mp_pose.PoseLandmark.LEFT_HIP,
            "RIGHT_HIP": mp_pose.PoseLandmark.RIGHT_HIP,
            "LEFT_KNEE": mp_pose.PoseLandmark.LEFT_KNEE,
            "RIGHT_KNEE": mp_pose.PoseLandmark.RIGHT_KNEE,
            "LEFT_ANKLE": mp_pose.PoseLandmark.LEFT_ANKLE,
            "RIGHT_ANKLE": mp_pose.PoseLandmark.RIGHT_ANKLE,
            "LEFT_HEEL": mp_pose.PoseLandmark.LEFT_HEEL,
            "RIGHT_HEEL": mp_pose.PoseLandmark.RIGHT_HEEL,
            "LEFT_FOOT_INDEX": mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            "RIGHT_FOOT_INDEX": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, 
        } 



        if results.pose_landmarks: # 座標が取得できた場合
            landmarks = []
            for name, index in pick_up.items():
                landmark = results.pose_landmarks.landmark[index]
                landmarks.extend([landmark.x, landmark.y, landmark.z,landmark.visibility])


            scaler = joblib.load("scaler.pkl")     
    
            
            # 取得したlandmarksをモデル入力用に前処理
            input_data = np.array(landmarks).reshape(1,-1)
            
            #データの標準化
            input_data_std = scaler.transform(input_data)
            print(input_data_std)

            # モデルで解析
            prediction = model.predict(input_data_std)
            pose_result = "good" if prediction[0] > 0.5 else "bad"

            # 結果をテンプレートに渡す
            return render_template('index.html', title="Pose Prediction", message="予測結果: ", result=f"{pose_result}:{prediction[0]}",uploaded_image_url=uploaded_image_url)
        else:
            return render_template('index.html', title="Pose Prediction", message="ポーズが検出されませんでした。")

    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
