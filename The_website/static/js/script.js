document.addEventListener("DOMContentLoaded", function() {   
    const paths = document.querySelectorAll('#mapsection path');
    const tooltip = document.getElementById('tooltip');

    // Add mouseover event listener to each path for tooltip
    paths.forEach((path) => {

        path.style.cursor = 'pointer'; // Change cursor to pointer

        // Add click event listener to each path for redirection
        path.addEventListener('click', function() {
            const index = Array.from(paths).indexOf(path);

            // Get the title attribute of the clicked path (region name)
            const regionName = this.getAttribute('title');

            // Save the region name to localStorage
            localStorage.setItem('selectedRegion', regionName);

            // Redirect to Page3.html
            window.location.href = '/Page3';
        });
    });
});
