//const webcamElement = document.getElementById('webcam');
//const classifier = knnClassifier.create();

//import * as lib from './lib.js';


const IMAGE_SIZE = 150;
const MODEL_PATH = '/model/model.json';

const REF_CLASSES = {
  0: 'Fresh Apple',
  1: 'Fresh Banana',
  2: 'Fresh Orange',
  3: 'Rotten Apple',
  4: 'Rotten Banana',
  5: 'Rotten Orange'
};



let model;
const warmup = async () => {
  // Load the model
  // Run prediction on sample image
  // Make prediction from local files, if uploaded

  const startTime1 = performance.now();
    
  console.log("load model...");
  model = await tf.loadGraphModel(MODEL_PATH);
  
  const elapsedTime1 = performance.now() - startTime1;
  console.log(`load model DONE in ${elapsedTime1} ms`);
  console.log("predict warmup sample...");
  
  const startTime2 = performance.now();
  
  const imgElement = document.getElementById('img');
  if (imgElement.complete && imgElement.naturalHeight !== 0) {
    predictImage(imgElement);
    imgElement.style.display = '';
  } else {
    imgElement.onload = () => {
      predictImage(imgElement)
      imgElement.style.display = '';
    }
  }
  
  const elapsedTime2 = performance.now() - startTime2;
  console.log(`predict warmup sample DONE in ${elapsedTime2} ms`);
    
  document.getElementById('container-file').style.display = '';
}


async function getClasses(prediction) {
  const values = await prediction.data();
  const topK = 2;

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });

  const topClassesAndProbs = [];
  for (let i = 0; i < topK; i++) {
    ix = valuesAndIndices[i].index;
    val = valuesAndIndices[i].value;
    topClassesAndProbs.push({
      className: REF_CLASSES[ix],
      probability: val
    })
  }
  return topClassesAndProbs;
};


async function predictImage(imgElement) {
  // Preprocess image
  // Return class which is the maximum prob
    
  const startTime = performance.now();
    
  const prediction = tf.tidy(() => {  
    let img = tf.browser.fromPixels(imgElement, 3)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE], false)
      .expandDims(0)
      .toFloat()
      .mul(1.0/255.0)

    return model.predict(img);
  });

  const elapsedTime = performance.now() - startTime;
  console.log(`predict img DONE in ${elapsedTime} ms`);
    
  const classes = await getClasses(prediction);
  
  showResults(imgElement, classes);
}


function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);
    
    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
//    probsElement.innerText = classes[i].probability.toFixed(3);
    probsElement.innerText = `${classes[i].probability.toFixed(4) * 100} %`;
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
}


const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = 256;
      img.height = 256;
      img.onload = () => predictImage(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});



const predictionsElement = document.getElementById('predictions');

warmup();