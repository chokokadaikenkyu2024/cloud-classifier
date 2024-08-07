let model;

window.onload = async () => {
  // モデルのロード
  model = await tf.loadLayersModel('/model/model.json');
  document.getElementById('classifyButton').addEventListener('click', classifyImage);
};

async function classifyImage() {
  const imageUpload = document.getElementById('imageUpload').files[0];
  if (!imageUpload) {
    alert('画像をアップロードしてください。');
    return;
  }

  // 画像を読み込み、グレースケールに変換してテンソルに変換
  const reader = new FileReader();
  reader.onload = async () => {
    const img = new Image();
    img.src = reader.result;
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0, img.width, img.height);

      // グレースケール変換
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        data[i] = avg;     // Red
        data[i + 1] = avg; // Green
        data[i + 2] = avg; // Blue
      }
      ctx.putImageData(imageData, 0, 0);

      const tensor = tf.browser.fromPixels(canvas)
        .resizeBilinear([50, 50])
        .expandDims()
        .toFloat()
        .div(tf.scalar(255.0));

      // 予測を実行
      const predictions = model.predict(tensor).dataSync();
      console.log(predictions); // デバッグ用

      // クラスラベルの決定
      const classLabels = ['Ac', 'As', 'Cb', 'Cc', 'Ci', 'Cs', 'Cu', 'Ns', 'Sc', 'St'];
      const maxIndex = predictions.indexOf(Math.max(...predictions));
      const label = classLabels[maxIndex];

      // 結果を表示
      document.getElementById('result').innerText = `この雲は: ${label} です。`;
    };
  };
  reader.readAsDataURL(imageUpload);
}
