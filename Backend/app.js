import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { log } from 'console';
import multer from 'multer';
import axios from 'axios';
import FormData from 'form-data';

dotenv.config();
const app = express();
const port = process.env.PORT || 3000;

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/api', (req, res) => {
    res.send('Hello from the Node.js proxy server.');
});

const handlePredict = async (req, res) => {
    log("=== Received prediction request ===");
    
    if (!req.file) {
        log("ERROR: No file uploaded");
        return res.status(400).json({ error: 'No file uploaded.' });
    }

    log("File info:", {
        name: req.file.originalname,
        size: req.file.size,
        mimetype: req.file.mimetype
    });

    try {
        // Create FormData with the uploaded file
        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        // Construct Python API URL based on your vercel.json routing
        const isProd = process.env.VERCEL === '1' || process.env.NODE_ENV === 'production';
        let pythonApiUrl;
        
        if (isProd) {
            // In production, both services are on the same Vercel domain
            const protocol = req.headers['x-forwarded-proto'] || 'https';
            const host = req.headers['x-forwarded-host'] || req.headers['host'];
            pythonApiUrl = `${protocol}://${host}/api/ai/predict`;
        } else {
            // Local development - you'll need to run the Python server separately
            pythonApiUrl = 'http://127.0.0.1:5000/api/ai/predict';
        }
        
        log("Calling Python AI service at:", pythonApiUrl);

        const response = await axios.post(pythonApiUrl, formData, {
            headers: {
                ...formData.getHeaders()
            },
            maxBodyLength: Infinity,
            timeout: 30000
        });

        log("SUCCESS - Received response from Python service:", response.data);
        res.status(200).json(response.data);

    } catch (error) {
        log("=== ERROR calling Python service ===");
        if (error.response) {
            log("Response error:", {
                status: error.response.status,
                statusText: error.response.statusText,
                data: error.response.data,
                url: error.config?.url
            });
        } else if (error.request) {
            log("Request error (no response):", error.message);
        } else {
            log("Setup error:", error.message);
        }
        
        res.status(500).json({ 
            error: 'Failed to get prediction from AI service.',
            details: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};

// Route matches your vercel.json: /api/predict goes to Backend/app.js
app.post('/api/predict', upload.single('file'), handlePredict);

app.listen(port, () => {
    log(`Backend server running on port ${port}`);
});

export default app;
