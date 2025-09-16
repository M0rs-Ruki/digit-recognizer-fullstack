// app.js
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { log } from 'console';
import multer from 'multer';
import axios from 'axios';
import FormData from 'form-data';

dotenv.config('./.env');
const app = express();
const port = process.env.PORT || 3000;

// Multer setup for handling file uploads in memory
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/', (req, res) => {
    res.send('Hello, World! This is the Node.js MERN Backend.');
});

// The new route to handle prediction requests
app.post('/predict', upload.single('file'), async (req, res) => {
    log("Received a request on /predict");

    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded.' });
    }

    try {
        // Create a FormData object to send to the Python server
        const formData = new FormData();
        // Append the file buffer from the uploaded file
        formData.append('file', req.file.buffer, { filename: req.file.originalname });

        log("Forwarding request to Python AI service...");

        // Make a POST request to the Python/Flask server
        const pythonApiUrl = 'http://127.0.0.1:5000/predict';
        const response = await axios.post(pythonApiUrl, formData, {
            headers: {
                ...formData.getHeaders()
            }
        });

        log("Received response from Python service:", response.data);

        // Extract the predicted_digit value from the Python service response
        const predictionData = { prediction: response.data.predicted_digit };

        // Send the prediction from the Python server back to the client
        res.status(200).json(predictionData);

    } catch (error) {
        log("Error calling Python service:", error.message);
        res.status(500).json({ error: 'Failed to get prediction from AI service.' });
    }
});

app.listen(port, () => {
    log(`Node.js Server is running: http://localhost:${port}`);
});