from imageai.Classification import ImageClassification
import os

exec_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(exec_path, 'mobilenet_v2-b0353104.pth'))
prediction.loadModel()

images = ['house.jpg', 'giraffe.jpg', 'godzilla.jpg']

for image_name in images:
    print(f'\n--- {image_name} ---')
    predictions, probabilities = prediction.classifyImage(
        os.path.join(exec_path, image_name), result_count=5
    )
    for eachPred, eachProb in zip(predictions, probabilities):
        print(f'{eachPred} : {eachProb}')