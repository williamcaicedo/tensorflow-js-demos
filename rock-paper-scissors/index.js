let mobilenetFeatureExtractor;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var rockSamples = 0, paperSamples = 0, scissorsSamples = 0, spockSamples = 0, lizardSamples = 0;
let isPredicting = false;

async function init() {
    await webcam.setup();
    mobilenetFeatureExtractor = await getMobilenetFeatureExtractor();
    //warm-up the model
    tf.tidy(() => mobilenetFeatureExtractor.predict(webcam.capture()));
    console.log('Feature extractor ready');
}

async function getMobilenetFeatureExtractor() {
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models' +
        '/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

async function train() {
    dataset.ys = null;
    dataset.encodeLabels(5);
    model = tf.sequential({
        layers: [
            tf.layers.flatten({ inputShape: mobilenetFeatureExtractor.outputs[0].shape.slice(1) }),
            tf.layers.dense({ units: 100, activation: 'relu' }),
            tf.layers.dense({ units: 5, activation: 'softmax' })
        ]
    });
    const optimizer = tf.train.adam(0.0001);
    model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
    let loss = 0;
    model.fit(dataset.xs, dataset.ys, {
        epochs: 10,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                loss = logs.loss.toFixed(5);
                console.log('LOSS: ' + loss);
            }
        }
    });

}

function doTraining() {
    train();
}

function startPredicting() {
    isPredicting = true;
    predict();
}

function stopPredicting() {
    isPredicting = false;
    predict();
}

function downloadModel() {
    model.save('downloads://my_model');
}

async function predict() {
    while (isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = webcam.capture();
            const activation = mobilenetFeatureExtractor.predict(img);
            const predictions = model.predict(activation);
            return predictions.as1D().argMax();
        });
        const classId = (await predictedClass.data())[0];
        switch (classId) {
            case 0:
                predictionText = 'I see Rock';
                break;
            case 1:
                predictionText = 'I see Paper';
                break;
            case 2:
                predictionText = 'I see Scissors';
                break;
            case 3:
                predictionText = 'I see Spock';
                break;
            case 4:
                predictionText = 'I see lizard';
        }
        document.getElementById('prediction').innerText = predictionText;
        predictedClass.dispose();
        // tf.nextframe returns a promise that resolve when a requestAnimationFrame has completed.
        // the idea here is not blocking the UI thread by awaiting whatever else has to be painted
        // on the browser to finish and then continuing with the infinite loop
        await tf.nextFrame();
    }
}

function handleButton(elem) {
    switch (elem.id) {
        case "0":
            rockSamples++;
            document.getElementById('rock_samples').innerText = 'Rock samples:' + rockSamples;
            break;
        case "1":
            paperSamples++;
            document.getElementById('paper_samples').innerText = 'Paper samples:' + paperSamples;
            break;
        case "2":
            scissorsSamples++;
            document.getElementById('scissors_samples').innerText = 'Scissors samples:' + scissorsSamples;
            break;
        case "3":
            spockSamples++;
            document.getElementById('spock_samples').innerText = 'Spock samples:' + spockSamples;
            break;
        case "4":
            lizardSamples++;
            document.getElementById('lizard_samples').innerText = 'Lizard samples:' + lizardSamples;
    }
    var label = parseInt(elem.id);
    const img = webcam.capture();
    dataset.addExample(mobilenetFeatureExtractor.predict(img), label);
}

init();