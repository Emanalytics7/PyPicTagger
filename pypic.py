import os
import csv
import backoff
from dotenv import load_dotenv
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

load_dotenv()
API_KEY = os.getenv('API')

MODEL_ID = 'general-image-recognition'
MODEL_VERSION_ID = 'aa7f35c01e0642fda5cf400f543e7c40'

class ClarifaiAPI:
    """A class to interact with Clarifai API."""

    def __init__(self, api):
        self.metadata =  (('authorization', 'Key ' + api),)
        self.channel = ClarifaiChannel.get_grpc_channel()
        self.stub = service_pb2_grpc.V2Stub(self.channel)

    @backoff.on_exception(backoff.expo,
                          Exception,
                          max_tries=8,
                          giveup=lambda e: not (e.status.code in [status_code_pb2.RESOURCE_EXHAUSTED, status_code_pb2.TOO_MANY_REQUESTS]))

    def classify_image(self, image_url):
        """Classify an image url and return predicted concpets."""

        if not image_url:
            print('Empty csv file.')
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
                metadata = self.metadata
            )

            if response.status.code != status_code_pb2.SUCCESS:
                raise Exception("Post model outputs failed, status:" + response.status.description)
            
            concepts = response.outputs[0].data.concepts
            return [(concept.name, concept.value) for concept in concepts]
        except Exception as e:
            print(f'Error processing {image_url}: {e}')
            return []

class CSVProcessor:
    """A class to process CSV files for image classification."""
    def __init__(self, clarifai_api):
        self.clarifai_api = clarifai_api
    
    def process_images(self, input_csv, ouput_csv):
        with open(input_csv, 'r') as infile, open(ouput_csv, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            writer.writerow(['IMAGE_URL', 'TAGS', 'Confidence Scores'])

            for row in reader:
                if not row:
                    continue # Skip empty rows

                image_url = row[0]
                if not image_url:
                    print('Found empty URL, skipping...')
                    continue

                concepts = self.clarifai_api.classify_image(image_url)
                tags = [concept[0] for concept in concepts]
                scores = [str(concept[1]) for concept in concepts]
                writer.writerow([image_url, ';'.join(tags), ';'.join(scores)])
              
# Example Usage
clarifai_api = ClarifaiAPI(API_KEY)
csv_processor = CSVProcessor(clarifai_api)
input_csv = 'input_images.csv'
output_csv = 'classified_images.csv'
csv_processor.process_images(input_csv, output_csv)