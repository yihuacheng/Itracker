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
  config = config["test"]
  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["load"]["model_name"] 
  
  loadpath = os.path.join(config["load"]["load_path"])
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  tests = os.listdir(labelpath)
  tests = [os.path.join(labelpath, j) for j in tests]
  
  savepath = os.path.join(loadpath, f"checkpoint")
  
  if not os.path.exists(os.path.join(loadpath, f"evaluation")):
    os.makedirs(os.path.join(loadpath, f"evaluation"))

  print("Read data")
  dataset = reader.txtload(tests, imagepath, 32, shuffle=False, num_workers=4, header=True)

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
    tabletaccs = 0
    phoneaccs = 0
    tcount = 0
    pcount = 0
    with torch.no_grad():
      with open(os.path.join(loadpath, f"evaluation/{saveiter}.log"), 'w') as outfile:
        outfile.write("name results gts\n")
        for j, (data, label) in enumerate(dataset):

          fimg = data["face"].to(device) 
          limg = data["left"].to(device) 
          rimg = data["right"].to(device) 
          grid = data["grid"].to(device)
          platforms = data["device"]
          names =  data["name"]

          img = {"left":limg, "right":rimg, "face":fimg, "grid":grid}
          gts = label.to(device)
           
          gazes = net(img)
          for k, gaze in enumerate(gazes):
            gaze = gaze.cpu().detach().numpy()

            if "iPad" in platforms[k]:
              tcount += 1
              tabletaccs += np.linalg.norm(gaze - gts.cpu().numpy()[k])
            if "iPhone" in platforms[k]:
              phoneaccs += np.linalg.norm(gaze - gts.cpu().numpy()[k])
              pcount += 1
            
            name = [names[k]]
            gaze = [str(u) for u in gaze] 
            gt = [str(u) for u in gts.cpu().numpy()[k]] 
            platform = [platforms[k]]
            log = name + [",".join(gaze)] + [",".join(gt)] + platform
            outfile.write(" ".join(log) + "\n")

        loger = f"[{saveiter}] Num: {tcount},{pcount},{tcount+pcount} tablet: {tabletaccs/tcount} Phone: {phoneaccs/pcount}"
        outfile.write(loger)
        print(loger)

