import * as mobilenet from "@tensorflow-models/mobilenet";
import * as tf from "@tensorflow/tfjs";

const FIVE_SECONDS_IN_MS = 5000;

class ImageClassifier {
  constructor() {
    this.loadModel();
  }

  async loadModel() {
    console.log("Loading model...");
    const startTime = performance.now();
    try {
      let json = chrome.runtime.getURL("src/jsmodels/filter_multi/model.json");
      chrome.storage.local.set({ model: json });
      this.model = await tf.loadLayersModel("jsmodels/filter_multi/model.json");
      const totalTime = Math.floor(performance.now() - startTime);
      console.log(`Model loaded and initialized in ${totalTime} ms...`);
    } catch (e) {
      console.error("Unable to load model", e);
    }
  }

  async analyzeImage(index, data) {
    if (!this.model) {
      console.log("Waiting for model to load...");
      setTimeout(() => {
        this.analyzeImage(index, data);
      }, FIVE_SECONDS_IN_MS);
      return;
    };
    console.log("Predicting...");

    let imageData = new ImageData(
      Uint8ClampedArray.from(data.rawImageData),
      data.width,
      data.height
    );
    imageData = tf.browser.fromPixels(imageData);
    let offset = tf.scalar(127.5);
    imageData = imageData.sub(offset).div(offset);
    imageData = tf.expandDims(imageData);

    let prediction = await this.model.predict(imageData);
    prediction.print();
    let result = await prediction.array();

    function clean_result(result){
      let result_class = ['cat', 'dog', 'other', 'snake', 'trypo'];
      let result_dic = {};
      for (let i=0; i<result.length; i++) {
        result_dic[result_class[i]] = result[i]
      }
      result_dic = Object.fromEntries(
          Object.entries(result_dic).sort(([,b],[,a]) => a-b)
      );
      return result_dic
    }
    return clean_result(result[0])
  }
}

const imageClassifier = new ImageClassifier();

chrome.runtime.onMessage.addListener(async (message, sender, senderResponse) => {
  // console.log(message.msg, message.index, message.url, message.data, sender.tab.id);
  if (message.msg == "image") {
    let results = await imageClassifier.analyzeImage(message.index, message.data);
    console.log(results);
    let returnMessage = {index: message.index, url:message.url, results: results};
    chrome.tabs.sendMessage(sender.tab.id, returnMessage);
  }

});