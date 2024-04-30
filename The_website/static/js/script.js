document.addEventListener("DOMContentLoaded", function() {   
    const paths = document.querySelectorAll('#mapsection path');
    const tooltip = document.getElementById('tooltip');

    // Add mouseover event listener to each path for tooltip
    paths.forEach((path) => {
        path.addEventListener('mouseover', function(event) {
            const regionName = event.target.getAttribute('title');
            tooltip.innerText = regionName;
            tooltip.style.opacity = 1;
            tooltip.style.left = event.pageX + 'px';
            tooltip.style.top = event.pageY + 'px';
        });

        // Add mouseout event listener to hide tooltip
        path.addEventListener('mouseout', function() {
            tooltip.style.opacity = 0;
        });

        path.style.cursor = 'pointer'; // Change cursor to pointer
    });

    // Add click event listener to each path for redirection and storing clicked region
    paths.forEach((path, index) => {
        path.addEventListener('click', function() {
            window.location.href = '/Page3';
            localStorage.setItem('clickedRegion', index);
        });
    });
});
