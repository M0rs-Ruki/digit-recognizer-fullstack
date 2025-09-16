import { useState } from 'react';
import { useForm } from 'react-hook-form';

// An SVG icon for the upload area for a cleaner look
const UploadIcon = () => (
  <svg
    className="w-12 h-12 mx-auto text-gray-400"
    stroke="currentColor"
    fill="none"
    viewBox="0 0 48 48"
    aria-hidden="true"
  >
    <path
      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

function App() {
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const { register, handleSubmit, reset, watch } = useForm();
  const uploadedImage = watch("digitImage");

  const handleImageChange = (event) => {
    const file = event.target.files[0];

    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please upload a valid image file (PNG, JPG, etc.).');
        setImagePreview(null);
        reset({ digitImage: null });
        return;
      }
      setError(null);
      setImagePreview(URL.createObjectURL(file));
      setPrediction(null);
    }
  };

  // --- THIS IS THE UPDATED SECTION ---
  const onSubmit = async (data) => {
    if (!data.digitImage || data.digitImage.length === 0) {
      setError('Please select an image first.');
      return;
    }

    setIsLoading(true);
    setPrediction(null);
    setError(null);

    // Create a FormData object to send the file to the backend
    const formData = new FormData();
    // The key 'file' must match what your Node.js backend expects: upload.single('file')
    formData.append('file', data.digitImage[0]);

    try {
      // Send the image to YOUR NODE.JS server (which runs on port 3000)
      const response = await fetch('http://localhost:3000/predict', {
        method: 'POST',
        body: formData, // No headers needed, browser sets it for FormData
      });

      if (!response.ok) {
        throw new Error('Server error. Make sure the backend services are running.');
      }

      const result = await response.json();

      if (result.error) {
        setError(result.error);
      } else {
        // The Python server sends back { "prediction": 7 }, so we access it here
        setPrediction(result.prediction);
      }

    } catch (error) {
      console.error('Error during prediction:', error);
      setError('Failed to get prediction. Check the console and ensure all servers are running.');
    } finally {
      // This will run whether the request succeeds or fails
      setIsLoading(false);
    }
  };
  
  const handleReset = () => {
    reset();
    setImagePreview(null);
    setPrediction(null);
    setError(null);
    setIsLoading(false);
  };
  

  return (
    <div className="bg-lime-200 min-h-screen flex items-center justify-center font-sans p-4">
      <div className="w-full max-w-2xl">
        
        {/* Main Card */}
        <div className="bg-white p-6 sm:p-8 border-2 border-black rounded-lg shadow-[8px_8px_0px_#000000] transition-all hover:shadow-[4px_4px_0px_#000000]">
          
          {/* Header */}
          <h1 className="text-3xl sm:text-4xl font-bold text-black text-center mb-2">
            Digit Recognition AI
          </h1>
          <p className="text-gray-600 text-center mb-6">
            Upload an image of a handwritten digit (0-9) and let the AI work its magic.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            
            {/* Left Side: Uploader */}
            <div className="flex flex-col">
              <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col h-full">
                <label
                  htmlFor="file-upload"
                  className="flex-grow flex flex-col justify-center items-center w-full p-4 border-2 border-dashed border-gray-400 rounded-md cursor-pointer hover:bg-gray-50 hover:border-black transition-colors"
                >
                  {imagePreview ? (
                    <img
                      src={imagePreview}
                      alt="Selected digit"
                      className="max-h-48 w-auto object-contain rounded-md"
                    />
                  ) : (
                    <div className="text-center">
                      <UploadIcon />
                      <span className="mt-2 block font-semibold text-gray-700">
                        Drop image here or <span className="text-blue-600">click to browse</span>
                      </span>
                    </div>
                  )}
                  <input
                    id="file-upload"
                    type="file"
                    className="hidden"
                    accept="image/*"
                    {...register('digitImage', { onChange: handleImageChange })}
                  />
                </label>
                
                <div className="flex items-center space-x-2 mt-4">
                  <button
                    type="submit"
                    disabled={isLoading || !imagePreview}
                    className="w-full font-bold text-lg text-black bg-yellow-400 border-2 border-black rounded-md px-6 py-3 transition-all
                                  hover:shadow-[4px_4px_0px_#000000] hover:bg-yellow-300
                                  active:shadow-[1px_1px_0px_#000000]
                                  disabled:bg-gray-300 disabled:shadow-none disabled:cursor-not-allowed"
                  >
                    {isLoading ? 'Recognizing...' : 'Recognize Digit'}
                  </button>
                   <button 
                      type="button" 
                      onClick={handleReset}
                      className="p-3 bg-red-500 border-2 border-black rounded-md transition-all hover:shadow-[4px_4px_0px_#000000] hover:bg-red-400 active:shadow-[1px_1px_0px_#000000]"
                      aria-label="Reset"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
                    </button>
                </div>
              </form>
            </div>

            {/* Right Side: Prediction */}
            <div className="flex flex-col items-center justify-center bg-gray-100 p-6 border-2 border-black rounded-lg">
              <h2 className="text-xl font-bold mb-4">Prediction</h2>
              <div className="w-full h-48 bg-white border-2 border-black rounded-md flex items-center justify-center">
                {error && <p className="text-red-600 font-semibold p-4 text-center">{error}</p>}
                
                {!error && isLoading && (
                    <div className="flex flex-col items-center">
                        <div className="w-8 h-8 border-4 border-dashed rounded-full animate-spin border-black"></div>
                        <p className="mt-3 text-gray-600">Analyzing image...</p>
                    </div>
                )}
                
                {!error && !isLoading && prediction !== null && (
                  <div className="text-center">
                    <p className="text-gray-600">Detected Digit:</p>
                    <p className="text-8xl font-black text-lime-600">{prediction}</p>
                  </div>
                )}

                {!error && !isLoading && prediction === null && (
                  <p className="text-gray-500 text-center p-4">Result will appear here</p>
                )}
              </div>
            </div>

          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-sm text-black mt-6">
            Powered by AI & <span className="font-bold">Neobrutalism</span>
        </p>

      </div>
    </div>
  );
}

export default App;

