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
    log("Received a request on /api/predict");

    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded.' });
    }

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);
        await fetch("/api/predict", {
        method: "POST",
        body: formData,
        });


        log("Error calling Python service:", error.response ? error.response.data : error.message);


        const isProd = process.env.VERCEL === '1' || process.env.NODE_ENV === 'production';
        const pythonBaseUrl = process.env.PYTHON_API_URL || 'http://127.0.0.1:5000';
        const forwardedProto = req.headers['x-forwarded-proto'] || 'https';
        const forwardedHost = req.headers['x-forwarded-host'] || req.headers['host'];
        const absolutePythonUrl = `${forwardedProto}://${forwardedHost}/api/ai/predict`;
        const pythonApiUrl = isProd ? absolutePythonUrl : `${pythonBaseUrl}/api/ai/predict`;        
        
        log("Calling Python AI service at:", pythonApiUrl);

        const response = await axios.post(pythonApiUrl, formData, {
            headers: {
                ...formData.getHeaders()
            },
            maxBodyLength: Infinity
        });

        log("Received response from Python service:", response.data);

        res.status(200).json(response.data);

    } catch (error) {
        log("Error calling Python service:", error.response ? error.response.data : error.message);
        res.status(500).json({ error: 'Failed to get prediction from AI service.' });
    }
};

app.post('/api/predict', upload.single('file'), handlePredict);
app.post('/', upload.single('file'), handlePredict);

app.listen(port, () => {
    log(`Node.js Server is running for local testing: http://localhost:${port}`);
});

export default app;
