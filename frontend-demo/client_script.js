// frontend-demo/client_script.js

// --- CONFIGURATION ---
const API_ENDPOINT = 'http://127.0.0.1:8000/api/utilities/extract-data';
// --- END CONFIGURATION ---

/**
 * Initiates the extraction process by gathering corrected data and sending the request to FastAPI.
 */
function startExtraction() {
    const transcriptInput = document.getElementById('transcript-input').value;
    const correctedName = document.getElementById('lead-name-correction').value.trim();
    const statusMessage = document.getElementById('status-message');
    const outputElement = document.getElementById('output');

    if (!transcriptInput) {
        statusMessage.textContent = 'Status: ERROR - Please paste a transcript first.';
        return;
    }

    statusMessage.textContent = 'Status: Sending request to FastAPI...';
    statusMessage.style.color = '#3498db';
    outputElement.textContent = 'Processing...';

    // --- 1. Apply Correction to Transcript (Simulated Final Clean Text) ---
    // In a real app, we'd only send the final clean name. Here, we simulate 
    // the system preparing the final transcript by substituting the name.
    
    let finalTranscript = transcriptInput;
    if (correctedName) {
        // Simple heuristic: replace the first major noun phrase after 'with' or 'for'
        finalTranscript = finalTranscript.replace(/with\s+([A-Za-z\s\.]+)|for\s+([A-Za-z\s\.]+)/, 
            match => match.replace(match.match(/([A-Za-z\s\.]+)$/)[0], correctedName));
    }


    // --- 2. Build Payload and Call API ---
    const payload = { transcript: finalTranscript };
    const startTime = performance.now();

    fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            // Handle HTTP errors (400, 500, etc.)
            return response.json().then(err => { throw new Error(err.detail || `HTTP Error: ${response.status}`); });
        }
        return response.json();
    })
    .then(data => {
        const endTime = performance.now();
        const latency = ((endTime - startTime) / 1000).toFixed(4); // Convert to seconds

        // --- 3. Display Final Results ---
        outputElement.textContent = JSON.stringify(data.data, null, 2);
        statusMessage.textContent = `Status: SUCCESS! Method: ${data.method}`;
        statusMessage.style.color = '#2ecc71';
        
        // Final Latency Report (includes network time)
        console.log(`Total Extraction Latency: ${latency}s`);
        
    })
    .catch(error => {
        statusMessage.textContent = `Status: EXTRACTION FAILED. ${error.message}`;
        statusMessage.style.color = '#e74c3c';
        outputElement.textContent = 'Error during API call. See console for details.';
        console.error('API Error:', error);
    });
}