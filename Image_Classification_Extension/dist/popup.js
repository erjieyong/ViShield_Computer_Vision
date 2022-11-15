document.addEventListener("DOMContentLoaded", function () {
  // Dog filter
  var checkbox_dog = document.querySelector('input[class="dog"]');
  chrome.storage.local.get("dogfilter", function (result) {
    let dogfilter = result.dogfilter;
    if (dogfilter == true) {
      checkbox_dog.checked = true;
    } else {
      checkbox_dog.checked = false;
    }
  });

  checkbox_dog.addEventListener("change", function () {
    if (checkbox_dog.checked) {
      // do this
      chrome.storage.local.set({ dogfilter: true }, function () {
        chrome.storage.local.get("dogfilter", function (result) {
          let dogfilter = result.dogfilter;
          console.log("Value is set to " + dogfilter);
        });
      });
    } else {
      // do that
      chrome.storage.local.set({ dogfilter: false }, function () {
        chrome.storage.local.get("dogfilter", function (result) {
          let dogfilter = result.dogfilter;
          console.log("Value is set to " + dogfilter);
        });
      });
    }
  });

  // cat filter
  var checkbox_cat = document.querySelector('input[class="cat"]');
  chrome.storage.local.get("catfilter", function (result) {
    let catfilter = result.catfilter;
    if (catfilter == true) {
      checkbox_cat.checked = true;
    } else {
      checkbox_cat.checked = false;
    }
  });

  checkbox_cat.addEventListener("change", function () {
    if (checkbox_cat.checked) {
      // do this
      chrome.storage.local.set({ catfilter: true }, function () {
        chrome.storage.local.get("catfilter", function (result) {
          let catfilter = result.catfilter;
          console.log("Value is set to " + catfilter);
        });
      });
    } else {
      // do that
      chrome.storage.local.set({ catfilter: false }, function () {
        chrome.storage.local.get("catfilter", function (result) {
          let catfilter = result.catfilter;
          console.log("Value is set to " + catfilter);
        });
      });
    }
  });

  // snake filter
  var checkbox_snake = document.querySelector('input[class="snake"]');
  chrome.storage.local.get("snakefilter", function (result) {
    let snakefilter = result.snakefilter;
    if (snakefilter == true) {
      checkbox_snake.checked = true;
    } else {
      checkbox_snake.checked = false;
    }
  });

  checkbox_snake.addEventListener("change", function () {
    if (checkbox_snake.checked) {
      // do this
      chrome.storage.local.set({ snakefilter: true }, function () {
        chrome.storage.local.get("snakefilter", function (result) {
          let snakefilter = result.snakefilter;
          console.log("Value is set to " + snakefilter);
        });
      });
    } else {
      // do that
      chrome.storage.local.set({ snakefilter: false }, function () {
        chrome.storage.local.get("snakefilter", function (result) {
          let snakefilter = result.snakefilter;
          console.log("Value is set to " + snakefilter);
        });
      });
    }
  });

  // trypo filter
  var checkbox_trypo = document.querySelector('input[class="trypo"]');
  chrome.storage.local.get("trypofilter", function (result) {
    let trypofilter = result.trypofilter;
    if (trypofilter == true) {
      checkbox_trypo.checked = true;
    } else {
      checkbox_trypo.checked = false;
    }
  });

  checkbox_trypo.addEventListener("change", function () {
    if (checkbox_trypo.checked) {
      // do this
      chrome.storage.local.set({ trypofilter: true }, function () {
        chrome.storage.local.get("trypofilter", function (result) {
          let trypofilter = result.trypofilter;
          console.log("Value is set to " + trypofilter);
        });
      });
    } else {
      // do that
      chrome.storage.local.set({ trypofilter: false }, function () {
        chrome.storage.local.get("trypofilter", function (result) {
          let trypofilter = result.trypofilter;
          console.log("Value is set to " + trypofilter);
        });
      });
    }
  });
});
