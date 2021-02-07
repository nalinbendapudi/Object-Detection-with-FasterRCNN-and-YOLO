import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt


def hello_single_stage_detector():
    print("Hello from single_stage_detector.py!")


def GenerateAnchor(anc, grid):
  """
  Anchor generator.

  Inputs:
  - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
    each point in the grid. anc[a] = (w, h) gives the width and height of the
    a'th anchor shape.
  - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
    center of each feature from the backbone feature map. This is the tensor
    returned from GenerateGrid.
  
  Outputs:
  - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
    anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
    centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
    boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
    and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
    corners of the box.
  """
  anchors = None
  ##############################################################################
  # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
  # generate all the anchor coordinates for each image. Support batch input.   #
  ##############################################################################
  # Replace "pass" statement with your code
  A,_ = anc.shape
  B,H,W,_ = grid.shape
  anchors = torch.zeros(B,A,H,W,4, device=grid.device, dtype = grid.dtype)
  for a in range(A):
    anchors[:,a,:,:,0] = grid[:,:,:,0] - anc[a,0]/2
    anchors[:,a,:,:,1] = grid[:,:,:,1] - anc[a,1]/2
    anchors[:,a,:,:,2] = grid[:,:,:,0] + anc[a,0]/2
    anchors[:,a,:,:,3] = grid[:,:,:,1] + anc[a,1]/2
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return anchors


def GenerateProposal(anchors, offsets, method='YOLO'):
  """
  Proposal generator.

  Inputs:
  - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
  - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
    anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
    (-0.5, 0.5).
  - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
  
  Outputs:
  - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
    proposal proposals[b, a, h, w].
  
  """
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  ##############################################################################
  # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
  # compute the proposal coordinates using the transformation formulas above.  #
  ##############################################################################
  # Replace "pass" statement with your code
  
  anchors_parametric = torch.zeros_like(anchors,dtype=anchors.dtype,device=anchors.device)
  anchors_parametric[:,:,:,:,:2] = (anchors[:,:,:,:,2:] + anchors[:,:,:,:,:2])/2  # x_c and y_c
  anchors_parametric[:,:,:,:,2:] =  anchors[:,:,:,:,2:] - anchors[:,:,:,:,:2]      # w and h

  proposals_parametric = torch.zeros_like(anchors,dtype=anchors.dtype,device=anchors.device)
  if method == 'YOLO':
    proposals_parametric[:,:,:,:,:2] = anchors_parametric[:,:,:,:,:2] + offsets[:,:,:,:,:2]
  else:
    proposals_parametric[:,:,:,:,:2] = anchors_parametric[:,:,:,:,:2] + offsets[:,:,:,:,:2]*anchors_parametric[:,:,:,:,2:]
  proposals_parametric[:,:,:,:,2:] = anchors_parametric[:,:,:,:,2:] * torch.exp(offsets[:,:,:,:,2:])

  proposals = torch.zeros_like(anchors,dtype=anchors.dtype,device=anchors.device)
  proposals[:,:,:,:,:2] = proposals_parametric[:,:,:,:,:2] - proposals_parametric[:,:,:,:,2:]/2
  proposals[:,:,:,:,2:] = proposals_parametric[:,:,:,:,:2] + proposals_parametric[:,:,:,:,2:]/2
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return proposals


def IoU(proposals, bboxes):
  """
  Compute intersection over union between sets of bounding boxes.

  Inputs:
  - proposals: Proposals of shape (B, A, H', W', 4)
  - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.
  
  Outputs:
  - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

  For this implementation you DO NOT need to filter invalid proposals or boxes;
  in particular you don't need any special handling for bboxxes that are padded
  with -1.
  """
  iou_mat = None
  ##############################################################################
  # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
  # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
  # However, you need to make sure to compute the IoU correctly (it should be  #
  # 0 in those cases.                                                          # 
  # You need to ensure your implementation is efficient (no for loops).        #
  # HINT:                                                                      #
  # IoU = Area of Intersection / Area of Union, where                          #
  # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
  # and the Area of Intersection can be computed using the top-left corner and #
  # bottom-right corner of proposal and bbox. Think about their relationships. #
  ##############################################################################
  # Replace "pass" statement with your code
  
  B,A,H,W,_ = proposals.shape
  B,N,_ = bboxes.shape
  M = A*H*W
  # print('B',B,'M',M,'N',N)

  proposals_n = proposals.reshape(B,M,4).unsqueeze(2)
  # print(proposals_n.shape)  # shape: (B,M,1,4)
  bboxes_n = bboxes[:,:,:4].unsqueeze(1)
  # print(bboxes_n.shape)  # shape: (B,1,N,4)

  tl_interxon = torch.max(proposals_n[:,:,:,:2],bboxes_n[:,:,:,:2])
  # print(tl_interxon.shape)  # shape: (B,M,N,2)
  br_interxon = torch.min(proposals_n[:,:,:,2:],bboxes_n[:,:,:,2:])
  # print(br_interxon.shape)  # shape: (B,M,N,2)
  area_interxon = torch.prod(br_interxon-tl_interxon, dim=3) * (tl_interxon<br_interxon).all(dim=3)
  # print(area_interxon.shape)  # shape: (B,M,N)

  area_proposal = torch.prod(proposals_n[:,:,:,2:]-proposals_n[:,:,:,:2], dim=3) 
  # print(area_proposal.shape)  # shape: (B,M,1)
  area_bbox = torch.prod(bboxes_n[:,:,:,2:]-bboxes_n[:,:,:,:2], dim=3)
  # print(area_bbox.shape)  # shape: (B,1,N)

  iou_mat = area_interxon / (area_proposal + area_bbox - area_interxon)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return iou_mat


class PredictionNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0 and num_anchors != 0)
    self.num_classes = num_classes
    self.num_anchors = num_anchors

    ##############################################################################
    # TODO: Set up a network that will predict outputs for all anchors. This     #
    # network should have a 1x1 convolution with hidden_dim filters, followed    #
    # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       #
    # finally another 1x1 convolution layer to predict all outputs. You can      #
    # use an nn.Sequential for this network, and store it in a member variable.  #
    # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                #
    # A=self.num_anchors and C=self.num_classes.                                 #
    ##############################################################################
    # Make sure to name your prediction network pred_layer.
    self.pred_layer = None
    # Replace "pass" statement with your code
    A=self.num_anchors
    C=self.num_classes
    self.pred_layer = nn.Sequential(
      nn.Conv2d(in_dim, hidden_dim, 1),
      nn.Dropout2d(p=drop_ratio),
      nn.LeakyReLU(),
      nn.Conv2d(hidden_dim, 5*A+C, 1)
    )
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors
  
  def _extract_class_scores(self, all_scores, anchor_idx):
    """
    Inputs:
    - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
      C classes at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
      which to extract classification scores

    Returns:
    - extracted_scores: Tensor of shape (M, C) giving the classification scores
      for each of the anchors specified by anchor_idx.
    """
    B, C, H, W = all_scores.shape
    A = self.num_anchors
    all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
    all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
    all_scores = all_scores.reshape(B * A * H * W, C)
    extracted_scores = all_scores[anchor_idx]
    return extracted_scores

  def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the network to predict outputs given features
    from the backbone network.

    Inputs:
    - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
      by the backbone network.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as positive. These are only given during training; at test-time
      this should be None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as negative. These are only given at training; at test-time this
      should be None.
    
    The outputs from this method are different during training and inference.
    
    During training, pos_anchor_idx and neg_anchor_idx are given and identify
    which anchors should be positive and negative, and this forward pass needs
    to extract only the predictions for the positive and negative anchors.

    During inference, only features are provided and this method needs to return
    predictions for all anchors.

    Outputs (During training):
    - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
      scores for positive anchors and negative anchors (in that order).
    - offsets: Tensor of shape (M, 4) giving predicted transformation for
      positive anchors.
    - class_scores: Tensor of shape (M, C) giving classification scores for
      positive anchors.

    Outputs (During inference):
    - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
      scores for all anchors.
    - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
      all all anchors.
    - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
      each spatial position.
    """
    conf_scores, offsets, class_scores = None, None, None
    ############################################################################
    # TODO: Use backbone features to predict conf_scores, offsets, and         #
    # class_scores. Make sure conf_scores is between 0 and 1 by squashing the  #
    # network output with a sigmoid. Also make sure the first two elements t^x #
    # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  #
    # and subtracting 0.5.                                                     #
    #                                                                          #
    # During training you need to extract the outputs for only the positive    #
    # and negative anchors as specified above.                                 #
    #                                                                          #
    # HINT: You can use the provided helper methods self._extract_anchor_data  #
    # and self._extract_class_scores to extract information for positive and   #
    # negative anchors specified by pos_anchor_idx and neg_anchor_idx.         #
    ############################################################################
    # Replace "pass" statement with your code
    
    B,in_dim,H,W = features.shape
    A=self.num_anchors
    C=self.num_classes
    
    network_output = self.pred_layer(features)  # shape: (B,5*A+C,H,W) 

    class_scores = network_output[:,5*A:,:,:]                   # shape: (B,C,H,W) 
    anchor_data  = network_output[:,:5*A,:,:].view(B,A,5,H,W)   # shape: (B,A,5,H,W) 
    conf_scores  = anchor_data[:,:,:1,:,:]                      # shape: (B,A,1,H,W)
    offsets      = anchor_data[:,:,1:,:,:]                      # shape: (B,A,4,H,W)

    conf_scores = torch.sigmoid(conf_scores)
    offsets[:,:,:2,:,:] = torch.sigmoid(offsets[:,:,:2,:,:]) - 0.5*torch.ones_like(offsets[:,:,:2,:,:])

    if pos_anchor_idx is None:  # inference
      conf_scores = conf_scores.squeeze(2)                      # shape : (B,A,H,W)
       
    else: # training
      M = pos_anchor_idx.shape
      
      offsets      = self._extract_anchor_data(offsets, pos_anchor_idx)        # shape: (M, 4)
      class_scores = self._extract_class_scores(class_scores, pos_anchor_idx)  # shape: (M, C)
      pos          = self._extract_anchor_data(conf_scores, pos_anchor_idx)    # shape: (M, 1)
      neg          = self._extract_anchor_data(conf_scores, neg_anchor_idx)    # shape: (M, 1)
      conf_scores  = torch.cat((pos, neg), dim=0)                              # shape: (2*M, 1)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return conf_scores, offsets, class_scores


class SingleStageDetector(nn.Module):
  def __init__(self):
    super().__init__()

    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    self.feat_extractor = FeatureExtractor()
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
  def forward(self, images, bboxes):
    """
    Training-time forward pass for the single-stage detector.

    Inputs:
    - images: Input images, of shape (B, 3, 224, 224)
    - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

    Outputs:
    - total_loss: Torch scalar giving the total loss for the batch.
    """
    # weights to multiple to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 1 # for offsets
    w_cls = 1 # for class_prob

    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of SingleStageDetector.                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, class_prob through the prediction network#
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,  #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression, and w_cls by ObjectClassification.                      #
    # HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedAnchors in this notebook #
    #       (A5-1) for a better performance than with the default value.         #
    ##############################################################################
    # Replace "pass" statement with your code
    
    C = self.num_classes
    B,N,_ = bboxes.shape
    A,_ = self.anchor_list.shape

    # 1. Image feature extraction
    features = self.feat_extractor(images)

    # 2a. Grid generation
    grid_list = GenerateGrid(B, dtype=images.dtype, device=images.device)
    self.anchor_list = self.anchor_list.to(images.dtype).to(images.device)
    # 2b. Anchors generation
    anc_list  = GenerateAnchor(self.anchor_list, grid_list)       # shape: (B,A,H,W,4)
    # 2c. anchors per img (required for classification loss)
    anc_per_img = torch.prod(torch.tensor(anc_list.shape[1:-1]))  # A*H*W

    # 3a. Compute IoU mat
    iou_mat = IoU(anc_list, bboxes)
    # 3b. Compute pos/neg anchors, GT_conf_scores, GT_offsets, GT_class
    activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, \
      activated_anc_coord, negative_anc_coord = ReferenceOnActivatedAnchors(anc_list, bboxes, grid_list, iou_mat, neg_thresh=0.2, method='YOLO')
    
    # 4. Compute conf_scores, offsets, class_prob through the prediction network
    conf_scores, offsets, class_prob = self.pred_network(features, activated_anc_ind, negative_anc_ind)

    # 5a. Compute classification loss for object category and regression loss for confidence & offset 
    conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
    reg_loss  = BboxRegression(offsets, GT_offsets)
    cls_loss  = ObjectClassification(class_prob, GT_class, B, anc_per_img, activated_anc_ind)
    # 5b. Compute total loss as a weighted sum
    total_loss = w_conf*conf_loss + w_reg*reg_loss + w_cls*cls_loss

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return total_loss
  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B  (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,         #
    # confidence scores `final_conf_scores`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
    # threshold `nms_thresh`.                                                    #
    # The class index is determined by the class with the maximal probability.   #
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
    # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
    ##############################################################################
    # Replace "pass" statement with your code
    with torch.no_grad():
      
      B,_,_,_ = images.shape
      A,_ = self.anchor_list.shape
      
      # 1. Image feature extraction
      features = self.feat_extractor(images)

      # 2a. Grid generation
      grid_list = GenerateGrid(B, dtype=images.dtype, device=images.device)
      self.anchor_list = self.anchor_list.to(images.dtype).to(images.device)
      # 2b. Anchors generation
      anc_list  = GenerateAnchor(self.anchor_list, grid_list)       # shape: (B,A,H,W,4)
      
      # 3a. Compute conf_scores, offsets, class_prob through the prediction network
      conf_scores, offsets, class_prob = self.pred_network(features) # shapes : (B,A,H,W) , (B,A,4,H,W) , (B,C,H,W)
      # 3b. Generate proposals
      offsets = offsets.permute(0,1,3,4,2).contiguous()   # shape: (B,A,H,W,4)
      proposals = GenerateProposal(anc_list, offsets)     # shape: (B,A,H,W,4)
      # 3c. Calculate predicted class
      classes = torch.argmax(class_prob, dim=1, keepdim=True) #shape: (B,1,H,W)
      # 3d. Uniformify the shapes of all variables
      conf_scores = conf_scores.reshape(B,-1)               # shape: (B,A*H*W)
      proposals   = proposals.reshape(B,-1,4)               # shape: (B,A*H*W,4)
      classes     = classes.repeat(1,A,1,1).reshape(B,-1)   # shape: (B,A*H*W)
      
      # 4. Loop over all images in the mini-batch, and filter by conf_scores and nms 
      for b in range(B):
        # 4a. Filter by conf_scores
        mask_conf = conf_scores[b] > thresh       # shape: (A*H*W,)
        proposals_b   = proposals[b,mask_conf,:]  # shape: (**,4)
        conf_scores_b = conf_scores[b,mask_conf]  # shape: (**,)
        class_b       = classes[b,mask_conf]      # shape: (**,)

        # 4b. Filter by nms
        mask_nms = torchvision.ops.nms(proposals_b, conf_scores_b, iou_threshold=nms_thresh)  # shape: (**,1)
        proposals_b   = proposals_b[mask_nms]                 # shape: (*,4)
        conf_scores_b = conf_scores_b[mask_nms].unsqueeze(1)  # shape: (*,1)
        class_b       = class_b[mask_nms].unsqueeze(1)        # shape: (*,1)

        # 4c. Append the tensors to final lists
        final_proposals.append(proposals_b)
        final_conf_scores.append(conf_scores_b)
        final_class.append(class_b)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return final_proposals, final_conf_scores, final_class


def nms(boxes, scores, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4
  - scores: scores for each one of the boxes, of shape N
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """

  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None
  #############################################################################
  # TODO: Implement non-maximum suppression which iterates the following:     #
  #       1. Select the highest-scoring box among the remaining ones,         #
  #          which has not been chosen in this step before                    #
  #       2. Eliminate boxes with IoU > threshold                             #
  #       3. If any boxes remain, GOTO 1                                      #
  #       Your implementation should not depend on a specific device type;    #
  #       you can use the device of the input if necessary.                   #
  # HINT: You can refer to the torchvision library code:                      #
  #   github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/nms_cpu.cpp  #
  #############################################################################
  # Replace "pass" statement with your code
  
  order = torch.argsort(scores, descending=True)
  areas = torch.prod(boxes[:, 2:] - boxes[:, :2], dim=1)
  keepList = []

  while order.shape[0] > 0:
    max_idx = order[0]
    order = order[1:]
    keepList.append(max_idx)
    
    max_box = boxes[max_idx]
    other_boxes = boxes[order]
    max_area = areas[max_idx]
    other_areas = areas[order]

    tl_interxon = torch.max(max_box[:2], other_boxes[:, :2]) 
    br_interxon = torch.min(max_box[2:], other_boxes[:, 2:])
    area_interxon = torch.prod(br_interxon - tl_interxon, dim=1) * (tl_interxon < br_interxon).all(dim=1)
    
    iou_mat = area_interxon / (max_area + other_areas - area_interxon)
    order_indices_to_keep = torch.where(iou_mat<=iou_threshold)
    order = order[order_indices_to_keep]

  if topk is not None:
    keepList = keepList[:topk]
  keep = torch.tensor(keepList).to(scores.device).to(torch.long) 

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return keep

def ConfScoreRegression(conf_scores, GT_conf_scores):
  """
  Use sum-squared error as in YOLO

  Inputs:
  - conf_scores: Predicted confidence scores
  - GT_conf_scores: GT confidence scores
  
  Outputs:
  - conf_score_loss
  """
  # the target conf_scores for negative samples are zeros
  GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                              torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
  conf_score_loss = torch.sum((conf_scores - GT_conf_scores)**2) * 1. / GT_conf_scores.shape[0]
  return conf_score_loss


def BboxRegression(offsets, GT_offsets):
  """"
  Use sum-squared error as in YOLO
  For both xy and wh

  Inputs:
  - offsets: Predicted box offsets
  - GT_offsets: GT box offsets
  
  Outputs:
  - bbox_reg_loss
  """
  bbox_reg_loss = torch.sum((offsets - GT_offsets)**2) * 1. / GT_offsets.shape[0]
  return bbox_reg_loss

