let model;

window.onload = async () => {
  // モデルのロード
  model = await tf.loadLayersModel('/model/model.json');
  document.getElementById('classifyButton').addEventListener('click', classifyImage);
};

async function classifyImage() {
  const imageUpload = document.getElementById('imageUpload').files[0];
  if (!imageUpload) {
    alert('Please upload an image first.');
    return;
  }

  // 画像を読み込み、テンソルに変換
  const reader = new FileReader();
  reader.onload = async () => {
    const img = new Image();
    img.src = reader.result;
    img.onload = () => {
      const tensor = tf.browser.fromPixels(img).resizeBilinear([50, 50]).expandDims().toFloat().div(tf.scalar(255.0));
      
      // 予測を実行
      const predictions = model.predict(tensor).dataSync();
      
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
