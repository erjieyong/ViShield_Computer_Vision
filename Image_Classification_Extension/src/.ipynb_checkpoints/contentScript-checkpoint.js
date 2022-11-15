let imgPath = chrome.runtime.getURL("/images/replaceImg.jpg");
// Size of the image expected by mobilenet.
const IMAGE_SIZE = 224;

// The minimum image size to consider classifying.  Below this limit the
// extension will refuse to classify the image.
const MIN_IMG_SIZE = 128;

const loadImageAndSendDataBack = async (src, sendResponse) => {
  // Load image (with crossOrigin set to anonymouse so that it can be used in a
  // canvas later).
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onerror = function (e) {
    console.warn(`Could not load image from external source ${src}.`);
    sendResponse({ rawImageData: undefined });
    return;
  };
  img.onload = async function (e) {
    if (
      (img.height && img.height > MIN_IMG_SIZE) ||
      (img.width && img.width > MIN_IMG_SIZE)
    ) {
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      // When image is loaded, render it to a canvas and send its ImageData back
      // to the service worker.
      const canvas = new OffscreenCanvas(img.width, img.height);
      // const canvas = document.createElement('canvas');
      const ctx = canvas.getContext("2d");
      // canvas.width=224;
      // canvas.height=224;
      ctx.drawImage(img, 0, 0, 224, 224);
      // document.body.appendChild(canvas);
      const imageData = ctx.getImageData(0, 0, img.width, img.height);

      // testing to check if the output after preprocessing is the same
      // only can be activated if we import tensor flow in contentscript and run yarn build
      // let imageData_new = new ImageData(
      //   Uint8ClampedArray.from(Array.from(imageData.data)), img.width, img.height);
      // imageData_new = tf.browser.fromPixels(imageData_new);
      // imageData_new.print();
      // const canvas1 = document.createElement('canvas');
      // tf.browser.toPixels(imageData_new,canvas1);
      // document.body.appendChild(canvas1);

      //to check if there's any difference when doing model predict here vs in background.js
      // let imageData_new1 = new ImageData(
      //   Uint8ClampedArray.from(Array.from(imageData.data)), img.width, img.height);
      // imageData_new1 = tf.browser.fromPixels(imageData_new1);
      // imageData_new1 = tf.expandDims(imageData_new1);

      // let json = chrome.runtime.getURL('src/model/model.json');
      // chrome.storage.local.set({'model': json});
      // const model = tf.loadLayersModel(json);
      // console.log(model);

      // let prediction = await model.predict(imageData);
      // prediction.print();

      sendResponse({
        rawImageData: Array.from(imageData.data),
        width: img.width,
        height: img.height,
      });
      return;
    }
    // Fail out if either dimension is less than MIN_IMG_SIZE.
    console.warn(
      `Image size too small. [${img.height} x ${img.width}] vs. minimum [${MIN_IMG_SIZE} x ${MIN_IMG_SIZE}]`
    );
    sendResponse({ rawImageData: undefined });
  };
  img.src = src;
};

let detectImg = function () {
  let images = document.getElementsByTagName("img");
  for (let i = 0; i < images.length; i++) {
    if (images[i].classList.contains("filtered")) {
      continue;
    } else {
      // add "filtered" class to all images that was checked before.
      images[i].classList.add("filtered");
      let url = images[i].src;
      console.log(`check on image ${i}, url: ${url}`);
      checkResult("image", i, url);
    }
  }
};


let checkResult = function (msg, index, url) {
  loadImageAndSendDataBack(url, function (data) {
    chrome.runtime.sendMessage({
      msg: msg,
      index: index,
      url: url,
      data: data,
    });
  });
};

detectImg();

//listen to model result from background.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log(message);
  chrome.storage.local.get(['dogfilter','catfilter','snakefilter','trypofilter'], function (filters) {
    let images = document.getElementsByTagName("img");
      for (let i = 0; i < images.length; i++) {
        if (images[i].src == message.url || images[i].srcset == message.url) {
          images[i].insertAdjacentHTML('afterend',`<div style = "position: absolute;bottom: 8px;right: 16px;">${Object.keys(message.results)[0]}, ${Object.values(message.results)[0].toFixed(4)}</div>`);
        }};
    // Dog filter
    if (filters.dogfilter && Object.keys(message.results)[0] == "dog") {
      //we replace image only if it is dog and url is the same. We have to do it this way out of detectImg because of alot of ajax websites which load and unload images dynamically. Hence, we have to search the dom each time
      for (let i = 0; i < images.length; i++) {
        if (images[i].src == message.url || images[i].srcset == message.url) {
          images[i].src = imgPath;
          images[i].srcset = imgPath;
        }
      }
    };
    // cat filter
    if (filters.catfilter && Object.keys(message.results)[0] == "cat") {
      //we replace image only if it is cat and url is the same. We have to do it this way out of detectImg because of alot of ajax websites which load and unload images dynamically. Hence, we have to search the dom each time
      let images = document.getElementsByTagName("img");
      for (let i = 0; i < images.length; i++) {
        if (images[i].src == message.url || images[i].srcset == message.url) {
          images[i].src = imgPath;
          images[i].srcset = imgPath;
          images[i].insertAdjacentHTML('afterend',`<div style = "position: absolute;bottom: 8px;right: 16px;">${Object.keys(message.results)[0]}, ${Object.values(message.results)[0].toFixed(4)}</div>`);
        }
      }
    };
    // snake filter
    if (filters.snakefilter && Object.keys(message.results)[0] == "snake") {
      //we replace image only if it is snake and url is the same. We have to do it this way out of detectImg because of alot of ajax websites which load and unload images dynamically. Hence, we have to search the dom each time
      let images = document.getElementsByTagName("img");
      for (let i = 0; i < images.length; i++) {
        if (images[i].src == message.url || images[i].srcset == message.url) {
          images[i].src = imgPath;
          images[i].srcset = imgPath;
          images[i].insertAdjacentHTML('afterend',`<div style = "position: absolute;bottom: 8px;right: 16px;">${Object.keys(message.results)[0]}, ${Object.values(message.results)[0].toFixed(4)}</div>`);
        }
      }
    };
    // trypo filter
    if (filters.trypofilter && Object.keys(message.results)[0] == "trypo") {
      //we replace image only if it is trypo and url is the same. We have to do it this way out of detectImg because of alot of ajax websites which load and unload images dynamically. Hence, we have to search the dom each time
      let images = document.getElementsByTagName("img");
      for (let i = 0; i < images.length; i++) {
        if (images[i].src == message.url || images[i].srcset == message.url) {
          images[i].src = imgPath;
          images[i].srcset = imgPath;
          images[i].insertAdjacentHTML('afterend',`<div style = "position: absolute;bottom: 8px;right: 16px;">${Object.keys(message.results)[0]}, ${Object.values(message.results)[0].toFixed(4)}</div>`);
        }
      }
    }
  });
});

// We run the mutationobserver so that our function detectImg can be called whenever there's ajax call
let observer = new MutationObserver(detectImg);

observer.observe(document.body, {
  childList: true,
  subtree: true,
});