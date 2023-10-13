document.getElementById("matchButton").addEventListener("click", () => {
    const carName = document.getElementById("carName").value;
    const speed = document.getElementById("speed").value;

    // You can make an AJAX request to the backend to send the carName and speed for processing.
    // Display the result in the #result div.
    // For simplicity, we'll just display a message here.

    const resultDiv = document.getElementById("result");
    resultDiv.textContent = `Matching ${carName} with a speed of ${speed}...`;
});
