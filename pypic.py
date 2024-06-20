import os
import csv
import backoff
import logging
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv('API')

MODEL_ID = 'general-image-recognition'
MODEL_VERSION_ID = 'aa7f35c01e0642fda5cf400f543e7c40'

class ClarifaiAPI:
    """A class to interact with Clarifai API."""

    def __init__(self, api):
        self.metadata = (('authorization', 'Key ' + api),)
        self.channel = ClarifaiChannel.get_grpc_channel()
        self.stub = service_pb2_grpc.V2Stub(self.channel)

    @backoff.on_exception(backoff.expo,
                          Exception,
                          max_tries=8,
                          giveup=lambda e: not (e.status.code in [status_code_pb2.RESOURCE_EXHAUSTED, status_code_pb2.TOO_MANY_REQUESTS]))
    def classify_image(self, image_url):
        """Classify an image url and return predicted concepts."""
        if not image_url:
            logger.warning('Empty image URL.')
            return []

        try:
            response = self.stub.PostModelOutputs(
                service_pb2.PostModelOutputsRequest(
                    model_id=MODEL_ID,
                    version_id=MODEL_VERSION_ID,
                    inputs=[
                        resources_pb2.Input(
                            data=resources_pb2.Data(
                                image=resources_pb2.Image(
                                url=image_url
                                )
                            )
                        )
                    ]
                ),
                metadata=self.metadata
            )

            if response.status.code != status_code_pb2.SUCCESS:
                raise Exception("Post model outputs failed, status:" + response.status.description)

            concepts = response.outputs[0].data.concepts
            return [(concept.name, concept.value) for concept in concepts]
        except Exception as e:
            logger.error(f'Error processing {image_url}: {e}')
            return []

class CSVProcessor:
    """A class to process CSV files for image classification."""
    def __init__(self, clarifai_api):
        self.clarifai_api = clarifai_api

    def validate_url(self, url):
        """Check if the URL is valid and accessible."""
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                return True
            else:
                logger.warning(f'Invalid URL {url} with status code {response.status_code}')
                return False
        except requests.RequestException as e:
            logger.warning(f'Error validating URL {url}: {e}')
            return False

    def process_image(self, image_url):
        """Process a single image URL."""
        if not self.validate_url(image_url):
            return [image_url, 'Invalid URL', 'N/A']

        concepts = self.clarifai_api.classify_image(image_url)
        tags = [concept[0] for concept in concepts]
        scores = [str(concept[1]) for concept in concepts]
        return [image_url, ';'.join(tags), ';'.join(scores)]

    def process_images(self, input_csv, output_csv):
        with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            writer.writerow(['IMAGE_URL', 'TAGS', 'Confidence Scores'])

            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(self.process_image, row[0]): row[0] for row in reader if row}
                for future in as_completed(future_to_url):
                    image_url = future_to_url[future]
                    try:
                        result = future.result()
                        writer.writerow(result)
                        logger.info(f'Processed {image_url}')
                    except Exception as e:
                        logger.error(f'Error processing {image_url}: {e}')

# Example Usage
if __name__ == "__main__":
    clarifai_api = ClarifaiAPI(API_KEY)
    csv_processor = CSVProcessor(clarifai_api)
    input_csv = 'input_images.csv'
    output_csv = 'classified_images.csv'
    csv_processor.process_images(input_csv, output_csv)
