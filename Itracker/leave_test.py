import model
import reader
import numpy as np
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import os
import copy
import importlib
def gazeto3d(gaze):
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt

def angular(gaze, label):
  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

if __name__ == "__main__":
  config = yaml.load(open(sys.argv[1]), Loader = yaml.FullLoader)
  dataloader = importlib.import_module("reader." + config['reader'])

  config = config["test"]
  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["load"]["model_name"] 
  
  loadpath = os.path.join(config["load"]["load_path"])
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  tests = os.listdir(labelpath)
  tests.sort()
  tests = tests.pop(int(sys.argv[2]))
  testpath = os.path.join(labelpath, tests)
  
  savepath = os.path.join(loadpath, f"checkpoint", tests)
  
  if not os.path.exists(os.path.join(loadpath, f"evaluation", tests)):
    os.makedirs(os.path.join(loadpath, f"evaluation", tests))

  print("Read data")
  dataset = dataloader.txtload(testpath, imagepath, 32, shuffle=False, num_workers=4, header=True)

  begin = config["load"]["begin_step"]
  end = config["load"]["end_step"]
  step = config["load"]["steps"]

  for saveiter in range(begin, end+step, step):
    print("Model building")
    net = model.ITrackerModel()
    statedict = torch.load(os.path.join(savepath, f"Iter_{saveiter}_{modelname}.pt"))

    net.to(device)
    net.load_state_dict(statedict)
    net.eval()

    print(f"Test {saveiter}")
    length = len(dataset)
    mmaccs = 0
    piaccs = 0
    count = 0
    with torch.no_grad():
      with open(os.path.join(loadpath, f"evaluation/{tests}/{saveiter}.log"), 'w') as outfile:
        outfile.write("name results gts\n")
        for j, (data, label) in enumerate(dataset):

          fimg = data["face"].to(device) 
          limg = data["left"].to(device) 
          rimg = data["right"].to(device) 
          grid = data["grid"].to(device)
          ratio = data["ratio"].cpu().numpy()[0]
          names =  data["name"]

          img = {"left":limg, "right":rimg, "face":fimg, "grid":grid}
          gts = label.to(device)
           
          gazes = net(img)
          for k, gaze in enumerate(gazes):
            gaze = gaze.cpu().detach().numpy()

            count += 1

            piaccs += np.linalg.norm(gaze*100/ratio - gts.cpu().numpy()[k]*100/ratio)

            mmaccs += np.linalg.norm(gaze*10 - gts.cpu().numpy()[k]*10)

            gaze = gaze/ratio
            gt =gts.cpu().numpy()[k]/ratio
          
            name = [names[k]]
            gaze = [str(u) for u in gaze] 
            gt = [str(u) for u in gt] 
            log = name + [",".join(gaze)] + [",".join(gt)] 
            outfile.write(" ".join(log) + "\n")

        loger = f"[{saveiter}] Num: {count} MMAcc: {mmaccs/count} PixelAcc: {piaccs/count}"
        outfile.write(loger)
        print(loger)

