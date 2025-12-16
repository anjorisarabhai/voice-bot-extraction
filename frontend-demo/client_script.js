// frontend-demo/client_script.js

// --- CONFIGURATION ---
const BASE_URL = 'http://127.0.0.1:8000/api/utilities';
const TRANSCRIBE_ENDPOINT = `${BASE_URL}/transcribe-audio`;
const EXTRACT_ENDPOINT = `${BASE_URL}/extract-data`;
// --- END CONFIGURATION ---

const statusMessage = document.getElementById('status-message');
const outputElement = document.getElementById('output');
const transcriptOutput = document.getElementById('transcript-output');
const correctionBox = document.getElementById('correction-box');
const correctedNameInput = document.getElementById('lead-name-correction');

let FINAL_TRANSCRIPT_TEXT = ''; 

/**
 * STEP 1: Uploads audio file and retrieves the text transcript from the backend ASR service.
 */
function startTranscription() {
    const audioFile = document.getElementById('audio-file-input').files[0];
    if (!audioFile) {
        statusMessage.textContent = 'Status: ERROR - Please select an audio file.';
        statusMessage.style.color = '#e74c3c';
        return;
    }

    statusMessage.textContent = 'Status: Uploading and transcribing audio (Running Whisper on Server)...';
    statusMessage.style.color = '#3498db';
    transcriptOutput.textContent = 'Processing audio...';
    correctionBox.style.display = 'none'; // Hide correction during ASR

    const formData = new FormData();
    formData.append('audio_file', audioFile);

    // Call the new ASR endpoint
    fetch(TRANSCRIBE_ENDPOINT, {
        method: 'POST',
        body: formData,
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.detail || response.statusText); });
        }
        return response.json();
    })
    .then(data => {
        FINAL_TRANSCRIPT_TEXT = data.transcript;
        transcriptOutput.textContent = FINAL_TRANSCRIPT_TEXT;
        statusMessage.textContent = `Status: Transcription Complete. Latency: ${data.asr_latency.toFixed(2)}s`;
        statusMessage.style.color = '#2ecc71';
        
        // Show correction box for HIL
        correctionBox.style.display = 'block';
    })
    .catch(error => {
        statusMessage.textContent = `Status: TRANSCRIPTION FAILED. Error: ${error.message}`;
        statusMessage.style.color = '#e74c3c';
        transcriptOutput.textContent = 'ASR Error. See console.';
        console.error('ASR API Error:', error);
    });
}

/**
 * STEP 2: Runs Extraction and then performs the CRITICAL HIL Database Validation check.
 */
async function startExtraction() {
    const correctedName = correctedNameInput.value.trim();
    
    if (!FINAL_TRANSCRIPT_TEXT) {
        statusMessage.textContent = 'Status: ERROR - No transcript available. Run ASR first.';
        statusMessage.style.color = '#e74c3c';
        return;
    }

    let finalTranscript = FINAL_TRANSCRIPT_TEXT;

    // Simulate Human-in-the-Loop Correction
    if (correctedName) {
        const regex = /(with|for)\s+([A-Z][a-z]+(\s+[A-Z][a-z]+)*)/i;
        if (regex.test(finalTranscript)) {
            finalTranscript = finalTranscript.replace(regex, `$1 ${correctedName}`);
        } else {
             finalTranscript = `Schedule a visit with ${correctedName} ${FINAL_TRANSCRIPT_TEXT}`;
        }
    }
    
    statusMessage.textContent = 'Status: Running Extraction Pipeline...';
    statusMessage.style.color = '#3498db';

    const payload = { transcript: finalTranscript };
    const startTime = performance.now();

    try {
        // 1. CALL EXTRACTION ENDPOINT
        let response = await fetch(EXTRACT_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || response.statusText);
        }
        
        let extractionResponse = await response.json();
        let extractedData = extractionResponse.data;
        const extractedLeadName = extractedData.lead_name;

        // 2. CALL VALIDATION ENDPOINT
        statusMessage.textContent = `Status: Extraction complete. Validating lead name: ${extractedLeadName}...`;

        let validationPayload = { "lead_name": extractedLeadName };
        let validationResponse = await fetch(`${BASE_URL}/search/validate_name`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(validationPayload)
        });

        if (!validationResponse.ok) {
            throw new Error("Validation API Failed.");
        }
        
        let validationResult = await validationResponse.json();
        
        const latency = ((performance.now() - startTime) / 1000).toFixed(4);
        
        // 3. FINAL HIL DECISION
        if (validationResult.status === "NEW_CONTACT") {
            // CRITICAL HIL STEP: Prompt for missing data
            extractedData.validation_status = "NEW_CONTACT_REQUIRED";
            extractedData.h_i_l_prompt = "Name not found in CRM. Please provide Email and Phone Number.";
            statusMessage.textContent = `Status: SUCCESS! METHOD: ${extractionResponse.method} - HIL PROMPT NEEDED. (API Time: ${latency}s)`;
            statusMessage.style.color = '#e67e22'; // Orange for HIL required
            
        } else {
            // Contact Exists - Final Success
            extractedData.validation_status = "CONTACT_EXISTS";
            extractedData.user_id = validationResult.user_id;
            statusMessage.textContent = `Status: SUCCESS! METHOD: ${extractionResponse.method} (API Time: ${latency}s)`;
            statusMessage.style.color = '#2ecc71';
        }

        outputElement.textContent = JSON.stringify(extractedData, null, 2);


    } catch (error) {
        statusMessage.textContent = `Status: EXTRACTION FAILED. ${error.message}`;
        statusMessage.style.color = '#e74c3c';
        outputElement.textContent = 'Error during API call. See console for details.';
        console.error('API Error:', error);
    }
}