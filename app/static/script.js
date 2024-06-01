document.getElementById('add-truck').addEventListener('click', () => {
    const truckInputs = document.getElementById('truck-inputs');
    const numTrucks = truckInputs.getElementsByClassName('truck-input').length;
    const newTruckInput = document.createElement('div');
    newTruckInput.classList.add('truck-input');
    newTruckInput.innerHTML = `
        <label for="truck-${numTrucks}">Truck ${numTrucks} (comma-separated values):</label>
        <input type="text" id="truck-${numTrucks}" name="truck-${numTrucks}">
    `;
    truckInputs.appendChild(newTruckInput);
});

document.getElementById('predict-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const truckInputs = document.getElementsByClassName('truck-input');
    const initialState = Array.from(truckInputs).map(truckInput => {
        const input = truckInput.getElementsByTagName('input')[0];
        return input.value.split(',').map(Number);
    });
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ initial_state: initialState }),
    });
    if (response.ok) {
        const results = await response.json();
        document.getElementById('results').innerText = JSON.stringify(results, null, 2);
    } else {
        document.getElementById('results').innerText = 'Error: ' + response.statusText;
    }
});
