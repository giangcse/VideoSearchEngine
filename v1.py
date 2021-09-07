"""Setup script for object_detection with TF2.0."""
import tensorflow as tf
import sys, os, time, pathlib, string, glob, hashlib, warnings, six
import numpy as np
import six.moves.urllib as urllib
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from elasticsearch import Elasticsearch
from xlwt import Workbook
sys.stdout.reconfigure(encoding='utf-8')
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
import logging
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

    
PATH_SAVED_MODEL = 'D:\\PT\\tensorflow2\\trained\\saved_model'
PATH_TO_LABELS = 'D:\\PT\\tensorflow2\\trained\\phuong-thao_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
MODEL_BASE = 'D:\\PT\\tensorflow2\\models'
sys.path.append(MODEL_BASE)
sys.path.append(MODEL_BASE + '\\research\\object_detection')
sys.path.append(MODEL_BASE + '\\research\\slim')
pipeline_config = 'D:\\PT\\tensorflow2\\trained\\pipeline_file.config'

# Load model, tính thời gian load model
def load_model():
    print('Loading model...', end='')
    start_time = time.time()
    model = tf.saved_model.load(str(PATH_SAVED_MODEL))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Load saved_model in {} seconds'.format(elapsed_time))
    return model
  
    
def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
        for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], 
            output_dict['detection_boxes'], 
            image.shape[0], image.shape[1])   
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def get_classes_name_and_scores(
        videoName, keyframeFile,
        boxes,
        classes,
        scores,
        category_index,
        max_boxes_to_draw=20,
        min_score_thresh=.4 # returns bigger than 70% precision
        ): 
    display_str = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            if classes[i] in six.viewkeys(category_index):
                display_str['classid'] = str(classes[i])
                display_str['name'] = category_index[classes[i]]['name']
                display_str['score'] = '{}%'.format(int(100 * scores[i]))
                display_str['s'] = int(100 * scores[i])
                stripped = category_index[classes[i]]['name']
                #"""#lưu dữ liệu vào elasticsearch
                es = Elasticsearch('localhost', http_auth=('elastic', 'changeme'), port=9200,)
                #print("Connected", es.info())
                vhash = hashlib.md5()
                vhash = hashlib.md5(stripped.encode())
                vhash.update(bytes(stripped, 'utf-8'))
                shash = '' + vhash.hexdigest()
                index_name = 'pt'
                hits = 0
                try:
                    #kiểm tra nội dung trước khi lưu vào Elasticsearch
                    if es.indices.exists(index = index_name):
                        search_param = {
                            "query": {
                                 "bool" : {
                                    "must" : [
                                        #{"match": {"frame": "000022"}} ,
                                        {"match": {"video": videoName}},
                                        {"match": {"vhash": shash}}
                                    ],
                                    #"filter": [{"match": {"vhash": vhash.hexdigest()}}]
                                }
                            }
                        }
                        response = es.search(index = index_name, body=search_param)
                        #print ('response:', response)
                        hits = len(response["hits"]["hits"])
                except Exception as ex:
                    print("Errorx:", ex)
                    pass
                try:
                    if hits>0:
                        print("Nội dung [" + shash + "] đã tồn tại cho video [" + videoName + "]")
                    else:
                        doc = {
                            'video': videoName,
                            'frame': os.path.basename(keyframeFile),
                            'content':  stripped,
                            'vhash':  shash
                        }
                        index_name = 'pt'
                        res = es.index(index = index_name, doc_type = 'samples', body = doc)
                        print(res)
                except Exception as ex:
                    print("Error:", ex)
                    pass
                #"""

    return display_str

def show_inference(model, image_path, videoName):
    print('Detecting...', end='')
    start_time = time.time()
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    '''vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    '''
    #display(Image.fromarray(image_np))
    framename = os.path.basename(os.path.splitext(image_path)[0])
    pred = get_classes_name_and_scores(
      videoName, framename,
      output_dict['detection_boxes'],
      (output_dict['detection_classes']).astype(int),
      output_dict['detection_scores'],
      category_index)
    print(pred)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Detect Image {} seconds'.format(elapsed_time))
    return pred

def main():
    print("Phien ban Tensorflow: ", tf.version.VERSION)
    print("Phien ban six: ", six.__version__)
    inputFolder = pathlib.Path('D:\\PT\\videos') #thu muc chua image, audio da cat
    videoExtension = '.indx';
    
    #TEST_IMAGE_PATHS
    detection_model = load_model()
    videoFiles = sorted(list(inputFolder.glob("*.indx")))
    # Workbook is created
    wb = Workbook()
    sheet1 = wb.add_sheet('vConfidence')
    sheet1.write(1, 0, 'STT')
    sheet1.write(1, 1, 'VIDEO NAME')
    sheet1.write(1, 2, 'TEXT')
    sheet1.write(1, 3, 'CONFIDENCE')
    rowIndex = 2
    countText = 0
    Avg4Image = 0.0
    SumConfidence = 0.0
    for videoFile in videoFiles:
        countText = 0 #đếm số dòng text trên mỗi video
        Avg4Image = 0.0 #trung bình confidence cho mỗi video
        SumConfidence = 0.0 #tổng confidence
        try:
            start_time = time.time()
            videoName = os.path.basename(os.path.splitext(videoFile)[0])
            #keyframesFolder = inputFolder + videoName + "_keyframes\\"
            keyframesFolder = os.path.join(inputFolder, videoName + "_keyframes\\") 
            print("=================[" + videoName + "]=======================")
            #kiểm tra thư mục keyframesFolder
            if not os.path.exists(keyframesFolder):
                print("Thư mục hình của video [" + videoName + "] không tồn tại, vui lòng kiểm tra lại!")
                continue
            else:
                #use for test
                for txtPath in glob.iglob(os.path.join(keyframesFolder, '*.txt')):
                    os.remove(txtPath)
                #khởi chạy spark
                PATH_TO_TEST_IMAGES_DIR = pathlib.Path(keyframesFolder)
                TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
                for image_path in TEST_IMAGE_PATHS:
                    prediction = show_inference(detection_model, image_path, videoName)
                    if len(prediction):
                        print(prediction["name"])
                        sheet1.write(rowIndex, 0, rowIndex - 1) #STT
                        sheet1.write(rowIndex, 1, videoName) #VIDEO NAME
                        sheet1.write(rowIndex, 2, prediction["name"]) #TEXT
                        sheet1.write(rowIndex, 3, prediction["score"]) #CONFIDENCE
                        rowIndex += 1
                        countText += 1
                        score = prediction["s"]
                        
                        SumConfidence += float(score)
                    #print(image_path)
            print("Thời gian thực thi cho video [" + videoName + "] là: %s giây" % round((time.time() - start_time), 2))  
        except Exception as ex:
            print("error: " + str(ex)) 
            pass
        if countText>0:
            Avg4Image = SumConfidence / countText
        else:
            Avg4Image = 0
        sheet1.write(rowIndex, 3, Avg4Image)
        rowIndex += 1
    wb.save('rcnn_confidence.xls')    
if __name__ == '__main__':
    main()