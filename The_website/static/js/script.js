// Wait for the DOM to be fully loaded
document.addEventListener("DOMContentLoaded", function() {   
    // Get all path elements in the SVG
    const paths = document.querySelectorAll('#mapsection path');

    // Add click event listener to each path
    paths.forEach((path, index) => {
        path.addEventListener('click', function() {
            // Redirect to page 3
            window.location.href = '/page3';

            // Save the clicked region (you can use localStorage, sessionStorage, or cookies)
            localStorage.setItem('clickedRegion', index);
        });
    });

    // Add an event listener for the form submission
    form.addEventListener('submit', function(event) {
        // Prevent the default form submission
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
