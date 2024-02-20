import subprocess
install_command = "pip install clarifai-grpc"
subprocess.call(install_command, shell=True)

#for installing the package clarifai-grpc using pip 


from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

def get_sentiment(text):
    # Your PAT (Personal Access Token) can be found in the portal under Authentification
    PAT = 'YOUR_PAT'
    # Specify the correct user_id/app_id pairings
    # Since you're making inferences outside your app's scope
    USER_ID = 'erfan'
    APP_ID = 'text-classification'
    # Change these to whatever model and text URL you want to use
    MODEL_ID = 'sentiment-analysis-twitter-roberta-base'
    MODEL_VERSION_ID = 'f7f3df02b79d4080a0233ec1fb6404bd'

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + PAT),)

    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            raw=text
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

    # Since we have one input, one output will exist here
    output = post_model_outputs_response.outputs[0]

    sentiment = []
    for concept in output.data.concepts:
        sentiment.append((concept.name, concept.value))

    return sentiment

text = input("Enter the text: ")
sentiment = get_sentiment(text)
print("Predicted concepts:")
for concept, value in sentiment:
    print("%s %.2f" % (concept, value))
