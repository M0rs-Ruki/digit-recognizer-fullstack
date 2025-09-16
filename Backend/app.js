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
const upload = multer({ 
    storage: storage,
    limits: {
        fileSize: 5 * 1024 * 1024 // 5MB limit
    }
});

app.use(cors({
    origin: true,
    credentials: false
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/api', (req, res) => {
    res.json({ 
        message: 'Node.js proxy server is running',
        timestamp: new Date().toISOString()
    });
});

app.get('/api/health', async (req, res) => {
    try {
        // Test connection to Python service
        const protocol = req.headers['x-forwarded-proto'] || 'https';
        const host = req.headers['x-forwarded-host'] || req.headers['host'];
        const pythonUrl = `${protocol}://${host}/api/ai`;
        
        const response = await axios.get(pythonUrl, { timeout: 10000 });
        res.json({
            nodejs: 'OK',
            python: response.data,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({
            nodejs: 'OK',
            python: 'ERROR: ' + error.message,
            timestamp: new Date().toISOString()
        });
    }
});

const handlePredict = async (req, res) => {
    const startTime = Date.now();
    log("=== Prediction Request Started ===");
    
    if (!req.file) {
        log("ERROR: No file uploaded");
        return res.status(400).json({ error: 'No file uploaded.' });
    }

    log("File details:", {
        name: req.file.originalname,
        size: req.file.size,
        mimetype: req.file.mimetype
    });

    try {
        // Create FormData properly
        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype || 'image/png'
        });

        // Construct Python API URL
        const protocol = req.headers['x-forwarded-proto'] || 'https';
        const host = req.headers['x-forwarded-host'] || req.headers['host'];
        const pythonApiUrl = `${protocol}://${host}/api/ai/predict`;
        
        log("Calling Python AI service at:", pythonApiUrl);

        const response = await axios.post(pythonApiUrl, formData, {
            headers: {
                ...formData.getHeaders(),
                'User-Agent': 'Node.js-Proxy'
            },
            maxBodyLength: Infinity,
            timeout: 30000,
            validateStatus: (status) => status < 500 // Don't throw on 4xx errors
        });

        const duration = Date.now() - startTime;
        log(`SUCCESS: Prediction completed in ${duration}ms`);
        log("Response:", response.data);

        res.status(200).json(response.data);

    } catch (error) {
        const duration = Date.now() - startTime;
        log(`ERROR: Request failed after ${duration}ms`);
        
        if (error.response) {
            log("Response error:", {
                status: error.response.status,
                statusText: error.response.statusText,
                data: error.response.data,
                headers: error.response.headers
            });
            res.status(error.response.status).json({
                error: 'Python service error',
                details: error.response.data
            });
        } else if (error.request) {
            log("Network error - no response received:", error.message);
            res.status(503).json({ 
                error: 'Cannot reach AI service',
                details: error.message
            });
        } else {
            log("Request setup error:", error.message);
            res.status(500).json({ 
                error: 'Request configuration error',
                details: error.message
            });
        }
    }
};

app.post('/api/predict', upload.single('file'), handlePredict);

app.get('/api/test-python', async (req, res) => {
    log("=== Python Test Request Started ===");
    try {
        const protocol = req.headers['x-forwarded-proto'] || 'https';
        const host = req.headers['x-forwarded-host'] || req.headers['host'];
        const pythonHealthUrl = `${protocol}://${host}/api/ai`;

        log(`Calling Python health check at: ${pythonHealthUrl}`);
        
        const response = await axios.get(pythonHealthUrl);
        
        log("Python health check response:", response.data);
        res.status(200).json({
            message: "Successfully called Python health check.",
            pythonResponse: response.data
        });

    } catch (error) {
        log("ERROR during Python health check:", error.message);
        res.status(500).json({
            error: "Failed to call Python health check.",
            details: error.message
        });
    }
});

app.listen(port, () => {
    log(`Backend server running on port ${port}`);
});

export default app;
