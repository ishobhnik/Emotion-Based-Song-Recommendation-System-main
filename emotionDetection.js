document.getElementById("emotion-form").addEventListener("submit", function(event) {
  event.preventDefault(); 
  let fileInput = document.getElementById("image-input");
  let file = fileInput.files[0];

  if (!file) {
      alert("Please select an image file.");
      return;
  }

  let formData = new FormData();
  formData.append("file", file);
    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      if (data.emotion) {
          document.getElementById("result").innerText = `Detected Emotion: ${data.emotion}`;
      } else {
          document.getElementById("result").innerText = `Error: ${data.error}`;
      }
  })
  .catch(error => {
      console.error("Error:", error);
      document.getElementById("result").innerText = "An error occurred.";
  });
});
