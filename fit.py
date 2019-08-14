# Silence warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import getopt
import os
import sys
import numpy as np
import SimpleITK as sitk
from progressbar import ProgressBar
from tensorflow.python.keras.models import load_model

SEGMENTER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'hematomasegmenter')
print(SEGMENTER_PATH)
sys.path.insert(1, SEGMENTER_PATH)

#from prostatesegmenter.data import ProstateData
#from prostatesegmenter.model import CNNModel
#from prostatesegmenter.nets import unet1
#from prostatesegmenter.segmenter import Segmenter

def predict(input_volume_path, output_mask_path):

    #input_volume_path = "/home/deepinfer/github/hematoma-segmenter/hematomasegmenter/data/10.nii.gz"

    print("Extracting data...")
    itkimage = sitk.ReadImage(input_volume_path)
    data = sitk.GetArrayFromImage(itkimage)

    print(np.unique(data))

    print("Pre-processing...")
    # fix orientation
    data = np.rot90(data, k=2, axes=(1, 2))
    data = np.flip(data, axis=2)
    data = np.flip(data, axis=1)

    # HU-clip
    limits = (0, 140)
    data[data < limits[0]] = limits[0]
    data[data > limits[1]] = limits[1]

    # scale -> [0, 1]
    data = data - np.amin(data)
    data = data / np.amax(data)

    # model path
    #model_path = input_volume_path.split("/")[:-2]
    model_path = "/home/deepinfer/github/hematoma-segmenter/hematomasegmenter/nets/unet_model.h5" #"/".join(model_path) + "/nets/" + "unet_model.h5"

    print("Loading model...")
    # load trained model
    model = load_model(model_path, compile=False)

    print("Predicting...")

    # split data into chunks and predict
    out_dim = (16, 512, 512)
    preds = np.zeros((int(np.ceil(data.shape[0] / out_dim[0])),) + out_dim + (2,)).astype(np.float32)
    for i in range(int(np.ceil(data.shape[0] / out_dim[0]))):
        data_out = np.zeros((1,) + out_dim + (1,), dtype=np.float32)
        tmp = data[i * out_dim[0]:i * out_dim[0] + out_dim[0]]
        data_out[0, :tmp.shape[0], ...,0] = tmp
        preds[i] = model.predict(data_out, verbose=1)

    label_nda = np.reshape(preds, (np.prod(preds.shape[:2]),) + preds.shape[2:])[:data.shape[0]]
    label_nda = label_nda[..., 1]

    print(np.unique(label_nda))

    print("Binarizing to produce label volume")
    th = 0.5
    label_nda[label_nda < th] = 0
    label_nda[label_nda >= th] = 1
    label_nda = label_nda.astype(np.uint8)
    

    print(np.unique(label_nda))

    label = sitk.GetImageFromArray(label_nda)
    itkimage = sitk.Cast(sitk.RescaleIntensity(itkimage), sitk.sitkUInt8)
    label.CopyInformation(itkimage)


    writer = sitk.ImageFileWriter()
    print("Writing to file...")
    #output_mask_path = "/home/deepinfer/" + "10_label.nrrd"
    print(input_volume_path)
    print(output_mask_path)
    writer.Execute(label, output_mask_path, False)

    #return out


#def predict(data):
#    pred = np.zeros((int(np.ceil(data.shape[0] / out_dim[0])),) + out_dim + (2,)).astype(np.float32)
#    for i in range(data.shape[0]):




def main(argv):
    InputVolume = ''
    OutputLabel = ''
    print("STARTING...")
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["InputVolume=", "OutputLabel="])
        print('---')
        print(args)
        print('---')
    except getopt.GetoptError:
        print('usage: fit.py -InputVolume <InputVolumePath> --OutputLabel <OutputLabelPath>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('fit.py -InputVolume <InputVolumePath> --OutputLabel <OutputLabelPath>')
            sys.exit()
        elif opt in ("-i", "--InputVolume"):
            InputVolume = arg
        elif opt in ("-o", "--OutputLabel"):
            OutputLabel = arg
    if InputVolume == '' or OutputLabel == '':
        print('usage: fit.py -InputVolume <InputVolumePath> -OutputLabel <OutputLabelPath>')
        sys.exit()

    print(InputVolume)
    print(OutputLabel)
    if os.path.isfile(InputVolume) and os.path.isdir(os.path.dirname(OutputLabel)):
        #print("Loading and preprocessing data.")
        #ds = ProstateData()
        #print("Making the model.")
        #model = unet1.model(weights=True, summary=False)
        #cnn = CNNModel(data_streamer=ds, model=model)
        #rows = 512
        #cols = 512
        #print("Predicting on data:")
        #
        print("Starting the segmenter.")
        predict(InputVolume, OutputLabel)
        #sg = Segmenter(cnn)
        #sg.segment_prostate_volume(InputVolume, OutputLabel, rows, cols)
    else:
        print("Make sure the input file exists and the output file directory is valid.")
        print("InputVolume: ", InputVolume)
        print("OutputLabel: ", OutputLabel)

if __name__ == "__main__":
    main(sys.argv[1:])