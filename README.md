# PyPicTagger

## Description
This Python script automates the process of classifying images. It reads image URLs from a CSV file, classifies them using the Clarifai API, and writes the results to another CSV file. This project showcases skills in API integration, error handling, and file operations in Python.

## Features
- **Automated Image Classification**: Streamlines the process of categorizing images based on their content.
- **API Integration**: Utilizes the Clarifai API for advanced image recognition.
- **Error Handling**: Implements robust error handling and retry strategies for API requests.
- **CSV Operations**: Reads from and writes to CSV files, enabling easy data manipulation and storage.

## Getting Started

### Dependencies
- Python 3.x
- `clarifai_grpc` for API access
- `python-dotenv` for environment variable management
- `backoff` for retry logic

### Setup
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Clarifai API key: `API_KEY=<Your_API_Key>`

### Usage
1. Prepare a CSV with image URLs in the first column.
2. Execute the script: `python pypic.py`
3. View results in `classified_images.csv`, including tags and confidence scores.

## Project Structure
- `pypic.py`: The main script that orchestrates the classification process.
- `.env`: An `.env` file.
- `requirements.txt`: Lists all the project dependencies.

## Contributing
We welcome contributions! If you're interested in improving this project, check the issues page for open tasks. To contribute:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

