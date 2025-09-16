import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { log } from 'console';
import multer from 'multer';
import axios from 'axios';
import FormData from 'form-data';

// Vercel doesn't use .env files in the same way. We use Environment Variables in the project settings.
dotenv.config();
const app = express();
const port = process.env.PORT || 3000;

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// This route can be removed for Vercel, but it's harmless to keep for testing.
app.get('/api', (req, res) => {
    res.send('Hello from the Node.js proxy server.');
});

// Support both local dev route '/api/predict' and Vercel function root '/'
const handlePredict = async (req, res) => {
    log("Received a request on /api/predict");

    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded.' });
    }

    try {
        const formData = new FormData();
        formData.append('file', req.file.buffer, { filename: req.file.originalname });

        log("Forwarding request to Python AI service...");

        // Compute Python API URL
        // - Production/Vercel: call the relative serverless route
        // - Local dev: use env PYTHON_API_URL or fallback to localhost
        const isProd = process.env.VERCEL === '1' || process.env.NODE_ENV === 'production';
        const pythonBaseUrl = process.env.PYTHON_API_URL || 'http://127.0.0.1:5000';
        const pythonApiUrl = isProd ? '/api/ai/predict' : `${pythonBaseUrl}/api/ai/predict`;
        
        log("Calling Python AI service at:", pythonApiUrl);

        const response = await axios.post(pythonApiUrl, formData, {
            headers: {
                ...formData.getHeaders()
            },
            maxBodyLength: Infinity
        });

        log("Received response from Python service:", response.data);

        // FIX 2: Pass the Python response directly through.
        // The Python API now correctly returns {'prediction': 7},
        // which is exactly what the frontend needs. No need to modify it.
        res.status(200).json(response.data);

    } catch (error) {
        log("Error calling Python service:", error.response ? error.response.data : error.message);
        res.status(500).json({ error: 'Failed to get prediction from AI service.' });
    }
};

app.post('/api/predict', upload.single('file'), handlePredict);
app.post('/', upload.single('file'), handlePredict);

// VERCEL NOTE: Vercel ignores this 'app.listen' block. It handles starting
// the server itself. This is only for running locally.
app.listen(port, () => {
    log(`Node.js Server is running for local testing: http://localhost:${port}`);
});

// Vercel needs this export to wrap the app in a serverless function
export default app;
