import copy
import math

import time 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import torch
import torchvision
import torchvision.transforms.functional as tvtf
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights,MaskRCNN_ResNet50_FPN_V2_Weights
# from torchvision.models.quantization import ResNet50_QuantizedWeights
# from torchvision.utils import make_grid
# from torchvision.io import read_image
from pathlib import Path


def load_img(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image(image):
    image = tvtf.to_tensor(image)
    image = image.unsqueeze(dim=0)
    return image


def display_image(image):
    fig, axes = plt.subplots(figsize=(12, 8))

    if image.ndim == 2:
        axes.imshow(image, cmap='gray', vmin=0, vmax=255)
    else:
        axes.imshow(image)

    plt.show()

    
def display_image_pair(first_image, second_image):
    #this funciton from Computer vision course notes 
    # When using plt.subplots, we can specify how many plottable regions we want to create through nrows and ncols
    # Here we are creating a subplot with 2 columns and 1 row (i.e. side-by-side axes)
    # When we do this, axes becomes a list of length 2 (Containing both plottable axes)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    
    # TODO: Call imshow on each of the axes with the first and second images
    #       Make sure you handle both RGB and grayscale images
    if first_image.ndim == 2:
        axes[0].imshow(first_image, cmap='gray', vmin=0, vmax=255)
    else:
        axes[0].imshow(first_image)

    if second_image.ndim == 2:
        axes[1].imshow(second_image, cmap='gray', vmin=0, vmax=255)
    else:
        axes[1].imshow(second_image)

    plt.show()

def get_detections(maskrcnn, imgs, score_threshold=0.5): #person, dog, elephan, zebra, giraffe, toilet
    ''' Runs maskrcnn over all frames in vid, storing the detections '''
    # Record how long the video is (in frames)
    det = []
    lbls = []
    scores = []
    masks = []
    
    for img in imgs:
        with torch.no_grad():
            result = maskrcnn(preprocess_image(img))[0]
    
        mask = result["scores"] > score_threshold

        boxes = result["boxes"][mask].detach().cpu().numpy()
        det.append(boxes)
        lbls.append(result["labels"][mask].detach().cpu().numpy())
        scores.append(result["scores"][mask].detach().cpu().numpy())
#         masks.append(result["masks"][mask].detach().cpu().numpy())
        masks.append(result["masks"][mask]) #I want this as a tensor
        
    # det is bounding boxes, lbls is class labels, scores are confidences and masks are segmentation masks
    return det, lbls, scores, masks

#draws the bounding boxes
COLOURS = [
    tuple(int(colour_hex.strip('#')[i:i+2], 16) for i in (0, 2, 4))
    for colour_hex in plt.rcParams['axes.prop_cycle'].by_key()['color']
]
def draw_detections(img, det, colours=COLOURS, obj_order = None):
    for i, (tlx, tly, brx, bry) in enumerate(det):
        if obj_order is not None and len(obj_order) < i:
            i = obj_order[i]
        i %= len(colours)
        c = colours[i]
        
        cv2.rectangle(img, (tlx, tly), (brx, bry), color=colours[i], thickness=2)

#annotate the class labels
weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
def annotate_class(img, det, lbls, conf=None, colours=COLOURS, class_map=weights.meta["categories"]):
    for i, ( tlx, tly, brx, bry) in enumerate(det):
        txt = class_map[lbls[i]]
        if conf is not None:
            txt += f' {conf[i]:1.3f}'
        # A box with a border thickness draws half of that thickness to the left of the 
        # boundaries, while filling fills only within the boundaries, so we expand the filled
        # region to match the border
        offset = 1
        
        cv2.rectangle(img, 
                      (tlx-offset, tly-offset+12),
                      (tlx-offset+len(txt)*12, tly),
                      color=colours[i%len(colours)],
                      thickness=cv2.FILLED)
        
        ff = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, txt, (tlx, tly-1+12), fontFace=ff, fontScale=1.0, color=(255,)*3)

def draw_instance_segmentation_mask(img, masks):
    ''' Draws segmentation masks over an img '''
    seg_colours = np.zeros_like(img, dtype=np.uint8)
    for i, mask in enumerate(masks):
        col = (mask[0, :, :, None] * COLOURS[i])
        seg_colours = np.maximum(seg_colours, col.astype(np.uint8))
    cv2.addWeighted(img, 0.75, seg_colours, 0.75, 1.0, dst=img)


def get_horizontal_centr_dist(boxes):
    pnts1 = np.array(get_center_of_boxes(boxes[0]))[:,0] #[:,0] dis guy is difrnt in get_vertical_centr_dist! 
    pnts2 = np.array(get_center_of_boxes(boxes[1]))[:,0]
    return pnts1[:,None] - pnts2[None]

def get_horizntl_dist_of_top_lft_crner(boxes):
    pnts1 = np.array(top_left_box_values(boxes[0]))[:,0]
    pnts2 = np.array(top_left_box_values(boxes[1]))[:,0]
    return pnts1[:,None] - pnts2[None]

def get_horizntl_dist_of_botm_rght_crner(boxes):
    pnts1 = np.array(bottom_right_box_values(boxes[0]))[:,0]
    pnts2 = np.array(bottom_right_box_values(boxes[1]))[:,0]
    print('sample pnts2 is __________________ ')
    print(pnts2)
    print('sample pnts1 is __________________')
    print(pnts1)
    return pnts1[:,None] - pnts2[None]

def get_vertical_centr_dist(boxes): 
    pnts1 = np.array(get_center_of_boxes(boxes[0]))[:,1]
    pnts2 = np.array(get_center_of_boxes(boxes[1]))[:,1]
    return pnts1[:,None] - pnts2[None]

def get_diffrnc_in_area_of_two_bxes(boxes):
    pnts1 = np.array(get_area(boxes[0]))
    pnts2 = np.array(get_area(boxes[1]))
    return abs(pnts1[:,None] - pnts2[None])
    


def get_center_of_boxes(boxes):
    print('boxes[0] is')
    print(boxes[0])
    print('boxes is ')
    print(boxes)
    points = []
    tlx,tly,brx,bry = boxes
    print('tlx,tly,brx,bry')
    print(tlx)
    print(tly)
    print(brx)
    print(bry)
    #for tlx, tly, brx, bry in boxes:
    cx = (tlx+brx)/2
    cy = (tly+bry)/2
    points.append([cx, cy])
    return points

def top_left_box_values(boxes):
    points = []
    tlx, tly, brx, bry  = boxes 
    #for tlx, tly, brx, bry in boxes:
    cx = (tlx+tlx)/2
    cy = (tly+tly)/2
    points.append((cx, cy))
    return points

def bottom_right_box_values(boxes):
    points = []
    tlx, tly, brx, bry  = boxes 
    #for tlx, tly, brx, bry in boxes:
    cx = (brx+brx)/2
    cy = (bry+bry)/2
    points.append((cx, cy))
    return points

def get_area(boxes):
    areas = []
    tlx, tly, brx, bry  = boxes 
    #for tlx, tly, brx, bry in boxes:
    cx = (brx-tlx)
    cy = (bry-tly)
    areas.append(abs(cx*cy))
    return areas 

def get_dist_to_centre_frm_top_lft(box, cntr):
    pnts = np.array(top_left_box_values(box))[:,0]
    return abs(pnts - cntr)

def get_dist_to_centre_frm_btm_rght(box, cntr):
    pnts = np.array(bottom_right_box_values(box))[:,0]
    return abs(pnts - cntr)

def annotate_class2(img, det, lbls,class_map, conf=None,  colours=COLOURS):
    for i, ( tlx, tly, brx, bry) in enumerate(det):
        txt = class_map[i]
        if conf is not None:
            txt += f' {conf[i]:1.3f}'
        # A box with a border thickness draws half of that thickness to the left of the 
        # boundaries, while filling fills only within the boundaries, so we expand the filled
        # region to match the border
        offset = 1
        
        cv2.rectangle(img, 
                      (tlx-offset, tly-offset+12),
                      (tlx-offset+len(txt)*12, tly),
                      color=colours[i%len(colours)],
                      thickness=cv2.FILLED)
        
        ff = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, txt, (tlx, tly-1+12), fontFace=ff, fontScale=1.0, color=(255,)*3)



#below prt seems good for viewing only dta frm two cameras_____________________BCKP_COPY_______________________________________________________________

# LEFT_URL = "http://192.168.142.67" # for left camera
# RIGHT_URL = "http://192.168.142.235" # for right camera

# #chatgpt stuff below. cv2.COLOR_BGR2GRAY was used by it, I use cv2.COLOR_BGR2RGB
# # Open the video stream from both cameras
# left_cap = cv2.VideoCapture(LEFT_URL + ":81/stream")
# right_cap = cv2.VideoCapture(RIGHT_URL + ":81/stream")

# # Check if both cameras are successfully opened
# if not left_cap.isOpened():
#     print("Can't open left camera")
#     exit()
# if not right_cap.isOpened():
#     print("Can't open right camera")
#     exit()

# # Start reading frames from both cameras
# while True:
#     # Read a frame from the left camera
#     left_ret, left_img = left_cap.read()
#     if not left_ret:
#         print("Can't receive left frame, exiting...")
#         break

#     # Read a frame from the right camera
#     right_ret, right_img = right_cap.read()
#     if not right_ret:
#         print("Can't receive right frame, exiting...")
#         break

#     # Convert the frames to grayscale (optional, depending on your needs)
#     cnvrt_frme_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
#     cnvrt_frme_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

#     # Display the frames
#     cv2.imshow('Left Camera', cnvrt_frme_left)
#     cv2.imshow('Right Camera', cnvrt_frme_right)

#     # Check for the 'q' key to exit the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video captures and close OpenCV windows
# left_cap.release()
# right_cap.release()
# cv2.destroyAllWindows()

#above prt seems good for viewing only dta frm two cameras___________________BCKP_COPY_________________________________________________________________


#below prt seems good for viewing only dta frm two cameras____________________________________________________________________________________

LEFT_URL = "http://192.168.142.67" # for left camera
RIGHT_URL = "http://192.168.142.235" # for right camera

#chatgpt stuff below. cv2.COLOR_BGR2GRAY was used by it, I use cv2.COLOR_BGR2RGB
# Open the video stream from both cameras
left_cap = cv2.VideoCapture(LEFT_URL + ":81/stream")
right_cap = cv2.VideoCapture(RIGHT_URL + ":81/stream")

# Check if both cameras are successfully opened
if not left_cap.isOpened():
    print("Can't open left camera")
    exit()
if not right_cap.isOpened():
    print("Can't open right camera")
    exit()

plt.ion()
import time

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
_ = model.eval()
# Start reading frames from both cameras
while True:
    # Read a frame from the left camera
    strt_two = time.time()
    left_ret, left_img = left_cap.read()
    if not left_ret:
        print("Can't receive left frame, exiting...")
        break

    # Read a frame from the right camera
    right_ret, right_img = right_cap.read()
    if not right_ret:
        print("Can't receive right frame, exiting...")
        break

    # Convert the frames to grayscale (optional, depending on your needs)
    cnvrt_frme_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    cnvrt_frme_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    end_two = time.time()
    imgs = [cnvrt_frme_left,cnvrt_frme_right]
    strt = time.time()
    det_zro = np.array([0,0,0,0],dtype=np.float32)
    det_one = np.array([0,0,0,0],dtype=np.float32)
    det_temp, lbls, scores, masks = get_detections(model,imgs)
    print('___det_temp___') # useful for seeing labels and detected boxes
    print(det_temp)
    print('__det_temp[0][0][0]___')
    print(det_temp[0][0][0])
    print('___lbls___')
    print(lbls)
    print(np.array(weights.meta["categories"])[lbls[0]])
    print(np.array(weights.meta["categories"])[lbls[1]])

    for a,entity_zro in enumerate(lbls[0]):
        if entity_zro == 44: # dis means entity named bottle is prsnt
            lbls[0] = [44] # writing lbls[0] as only a single entity found
            print('a is ',a)
            det_zro[0] = det_temp[0][a][0]
            det_zro[1] = det_temp[0][a][1]
            det_zro[2] = det_temp[0][a][2]
            det_zro[3] = det_temp[0][a][3]
            print('det_zro is ',det_zro)
            break
        else:
            lbls[0] = []
    for b,entity_one in enumerate(lbls[1]):
        if entity_one == 44: # dis means entity named bottle is prsnt
            lbls[1] = [44] # writing lbls[0] as only a single entity found
            det_one[0] = det_temp[1][b][0]
            det_one[1] = det_temp[1][b][1]
            det_one[2] = det_temp[1][b][2]
            det_one[3] = det_temp[1][b][3]
            print('det_one is ',det_one)
            break
        else:
            lbls[1] = []
    if lbls[0] == [] or lbls[1] == []:
        pass
    else:
        end = time.time()
        print('print(strt - end) is ')
        print(strt - end)
        print('strt_two - end_two is ')
        print(strt_two - end_two)
        sz1 = right_img.shape[1]
        centre = sz1/2
        print('centre is ')
        print(centre)
        det=[det_zro,det_one] # dis means left, right
        dists_tl = get_horizntl_dist_of_top_lft_crner(det)
        print('dists_tl is ________________________________')
        print(dists_tl)
        dists_br = get_horizntl_dist_of_botm_rght_crner(det)
        print('dists_br is ________________________________')
        print(dists_br)
        final_dists = []
        dctl = get_dist_to_centre_frm_top_lft(det[0],cntr = centre)
        print('dctl is _____________________')
        print(dctl)
        dcbr = get_dist_to_centre_frm_btm_rght(det[0], cntr = centre) 
        print('dcbr is ______________________')
        print(dcbr)
        for i in range(1):
            if dctl[0] < dcbr[0]:# dis means take d box which is closer to centre and find tl/br distance acrdngly
                final_dists.append((dists_tl[0][0],np.array(weights.meta["categories"])[lbls[0]][0]))
            else:
                final_dists.append((dists_br[0][0],np.array(weights.meta["categories"])[lbls[0]][0]))
        print('final_dists is ')
        print(final_dists)
        fd = [i for (i,j) in final_dists]
        tantheta = 0.5
        fl = 2.54
        dists_away = (5.5/2)*sz1*(1/tantheta)/np.array((fd))+fl
        for i in range(len(dists_away)):
            print(f'{np.array(weights.meta["categories"])[lbls[0]][0]} is {dists_away[i]:.1f}cm away')
        # t1 = [list(tracks[1]), list(tracks[0])]
        # frames_ret = []
        # for i, imgi in enumerate(imgs):
        #     img = imgi.copy()
        #     deti = det[i].astype(np.int32)
        #     draw_detections(img,deti[list(tracks[i])], obj_order=list(t1[1]))
        #     annotate_class2(img,deti[list(tracks[i])],lbls[i][list(tracks[i])],cat_dist)
        #     frames_ret.append(img)
        # cv2.imshow("left_eye", cv2.cvtColor(frames_ret[0],cv2.COLOR_RGB2BGR))
        # cv2.imshow("right_eye", cv2.cvtColor(frames_ret[1],cv2.COLOR_RGB2BGR))


        # Display the frames
        # cv2.imshow('Left Camera', cnvrt_frme_left)
        # cv2.imshow('Right Camera', cnvrt_frme_right)


        #display_image_pair(left_img,right_img) dis step isn't needed

        # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video captures and close OpenCV windows
left_cap.release()
right_cap.release()
cv2.destroyAllWindows()

#above prt seems good for viewing only dta frm two cameras____________________________________________________________________________________

# sz1 = right_img.shape[1]
# #print('sz1 of right_img,(N) is ') dis is size of image (N) needed in our distnc formla 
# #print(sz1)
# sz2 = right_img.shape[0]

# #display_image_pair(left_img,right_img) dis step isn't needed
# imgs = [left_img,right_img]
# left_right = [preprocess_image(d).squeeze(dim=0) for d in imgs]

# model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
# _ = model.eval()

# det, lbls, scores, masks = get_detections(model,imgs)


# fig, axes = plt.subplots(1, 2, figsize=(12, 8))
# # imgs1 = imgs.copy()

# for i, imgi in enumerate(imgs):
#     img = imgi.copy()
#     deti = det[i].astype(np.int32)
#     draw_detections(img,deti)
#     masksi = masks[i].detach().cpu().numpy()
#     annotate_class(img,deti,lbls[i])
# #     draw_instance_segmentation_mask(img, masksi)
#     axes[i].imshow(img)
#     axes[i].axis('off')
#     axes[i].set_title(f'Frame #{i}')

# plt.show() #not showing anythn curntly ,shows lft cmra on left and rght cmra on rght
#get centr, top left and bottom right of boxes


# #get all distances from every object box to every other object box
# #left image is boxes[0]
# #right image is boxes[1]

# #do broad casting.
# #in python, col vector - row vector gives matrix:
# # [a] - [c,d] = [a-c, a-d]
# # [b]           [b-c, b-d] __i dont know where dis stuff is cnctd to. (<-IHW)


# ## get distance bentween corner and centre

# centre = sz1/2


# #create the tracking cost function.
# #consists of theree parts.
# #  1. The vertical move up and down of object centre of mass. Scale this up because we do not expect this to be very much.
# #  2. The move left or right by the object. We only expect it to move right (from the left eye image). So penalise if it moves left.
# #  3. The difference in area of pixels. Area of image is width x height, so divide by height, there for this will have max value of width


# def get_horiz_dist(masks, prob_thresh = 0.7):
#     # gets the horizontal distance between the centre of mass for each object
#     #left masks
#     mask_bool = masks[0] > prob_thresh
#     mask_bool = mask_bool.squeeze(1)
#     #right masks
#     mask_bool2 = masks[1] > prob_thresh
#     mask_bool2 = mask_bool2.squeeze(1)
    
#     #left params
#     #com1 is center of mass of height
#     #com2 is center of mass of width
#     mask_size = (mask_bool).sum(dim=[1,2])
#     mask_com_matrix_1 = torch.tensor(range(mask_bool.shape[1]))
#     com1 = ((mask_com_matrix_1.unsqueeze(1))*mask_bool).sum(dim=[1,2])/mask_size
#     mask_com_matrix_2 = torch.tensor(range(mask_bool.shape[2]))
#     com2 = ((mask_com_matrix_2.unsqueeze(0))*mask_bool).sum(dim=[1,2])/mask_size

#     left_params = torch.stack((com1, com2, mask_size)).transpose(1,0)
    
#     #get right params
#     mask_size2 = (mask_bool2).sum(dim=[1,2])
#     mask_com_matrix_12 = torch.tensor(range(mask_bool2.shape[1]))
#     com12 = ((mask_com_matrix_12.unsqueeze(1))*mask_bool2).sum(dim=[1,2])/mask_size2
#     mask_com_matrix_22 = torch.tensor(range(mask_bool2.shape[2]))
#     com22 = ((mask_com_matrix_22.unsqueeze(0))*mask_bool2).sum(dim=[1,2])/mask_size2

#     right_params = torch.stack((com12, com22, mask_size2)).transpose(1,0)
    
#     #calculate cost function
#     cost = (left_params[:,None] - right_params[None])
#     return cost[:,:,1]

# cost = get_cost(det, lbls = lbls)

# tracks = scipy.optimize.linear_sum_assignment(cost)


# h_d = [[np.array(weights.meta["categories"])[lbls[0]][i],np.array(weights.meta["categories"])[lbls[1]][j]] for i, j in zip(*tracks)]


# dists_tl =  get_horizntl_dist_of_top_lft_crner(det)
# dists_br =  get_horizntl_dist_of_botm_rght_crner(det)

# final_dists = []
# dctl = get_dist_to_centre_frm_top_lft(det[0],cntr = centre) # dis guy needs modification. 'centre' needs to be explicitly given like get_dist_to_centre_frm_top_lft(det[0],cntr = centre) where centre = 8  for instance.!! WARNING !!
# dcbr = get_dist_to_centre_frm_btm_rght(det[0]) # like above, dis guy needs modification too!

# for i, j in zip(*tracks):
#     if dctl[i] < dcbr[i]:
#         final_dists.append((dists_tl[i][j],np.array(weights.meta["categories"])[lbls[0]][i]))
        
#     else:
#         final_dists.append((dists_br[i][j],np.array(weights.meta["categories"])[lbls[0]][i]))
        

# print(final_dists)




# above code works decently for getting depth for two ESP32 cameras. 

