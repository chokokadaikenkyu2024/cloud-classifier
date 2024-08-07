const modelUrl = 'https://chokocloudtest.netlify.app/model/model.json'; // GitHub PagesのURL
let model;

const classNames = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Cu", "Ns", "Sc", "St"];

async function loadModel() {
  try {
    model = await tf.loadLayersModel(modelUrl);
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
  }
}

function processImage(image) {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = image.width;
  canvas.height = image.height;
  ctx.drawImage(image, 0, 0);
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  
  // グレースケール変換
  for (let i = 0; i < imgData.data.length; i += 4) {
    const avg = (imgData.data[i] + imgData.data[i + 1] + imgData.data[i + 2]) / 3;
    imgData.data[i] = imgData.data[i + 1] = imgData.data[i + 2] = avg;
  }
  ctx.putImageData(imgData, 0, 0);
  
  const tensor = tf.browser.fromPixels(canvas).toFloat().div(tf.scalar(255)).expandDims();
  
  model.predict(tensor).array().then(predictions => {
    console.log(predictions); // デバッグ用
    const prediction = predictions[0];
    const maxIndex = prediction.indexOf(Math.max(...prediction));
    const result = classNames[maxIndex];
    document.getElementById('result').textContent = `Predicted class: ${result}`;
  }).catch(error => {
    console.error('Error making prediction:', error);
  });
}

document.getElementById('upload').addEventListener('change', event => {
  const file = event.target.files[0];
  const image = new Image();
  image.onload = () => processImage(image);
  image.src = URL.createObjectURL(file);
});

loadModel();
