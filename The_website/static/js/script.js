// Wait for the document to be fully loaded
document.addEventListener("DOMContentLoaded", function() {
    var nord = document.querySelector("#Nordjylland");
    var midt = document.querySelector("#Midtjylland");
    console.log(nord);
    console.log(midt);

    // Get the prediction form element
    const form = document.getElementById("prediction-form");
    
    // Add an event listener for the form submission
    form.addEventListener("submit", function(event) {
        event.preventDefault();

        // Gather form data
        const formData = new FormData(form);
        
        // Send a POST request to the /id/ endpoint with the form data
        fetch("/id/", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Display the prediction result on the page
            const predictionOutput = document.getElementById("prediction-output");
            predictionOutput.textContent = "Prediction: " + data.prediction;
        })        
        .catch(error => {
            // Handle any errors that occur during the fetch
            console.error("Error:", error);
        });
    });


});
