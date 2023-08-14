import os

import cv2
import torch
import os.path
import numpy as np
from PIL import Image
import mediapipe as mp
from typing import Mapping
import albumentations as A
from skimage.io import imread
import albumentations.pytorch
import torch.nn.functional as F
from .base_dataset import BaseDataset


class RenderDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.
        Modified to 68 keypoints version

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt)
        self.state = self.opt.dataset_type
        self.dataset_name = opt.dataset_names[0]
        self.use_da = opt.use_da
        self.use_hf = opt.use_hf
        self.tpos_param = opt.temporal_encoding_param

        self.part_list = [[list(range(0, 17))],                                 # contour
             [list(range(17, 22))],                                             # right eyebrow
             [list(range(22, 27))],                                             # left eyebrow
             [list(range(27, 31)), list(range(31, 36))],                        # nose
             [list(range(36, 42)), [41, 36]],                                   # right eye
             [list(range(42, 48)), [47, 42]],                                   # left eye
             [list(range(48, 55)), list(range(55, 60)), [59, 48], [55, 54]],    # mouth
             [list(range(60, 65)), list(range(65, 68)), [67, 60], [65, 64]]     # tongue
            ]
        self.mouth_outer = [48, 49, 50, 51, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 48]
                
        # only load in train mode
          
        self.dataset_root = os.path.join(self.root, self.dataset_name, 'render')
        if self.state == 'Train':
            self.item_list = self.opt.train_dataset_names
        elif self.state == 'Val':
            self.item_list = self.opt.validate_dataset_names
        else:
            self.item_list = []

        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_connections = mp.solutions.face_mesh_connections
        
        self.data_root_tar    = os.path.join(self.dataset_root, 'pix2pix/target')
        self.data_root_feat   = os.path.join(self.dataset_root, 'pix2pix/feature')
        self.data_root_cand   = os.path.join(self.dataset_root, 'pix2pix/candidate')
        self.data_root_weight = os.path.join(self.dataset_root, 'pix2pix/mouth')

        self.frame_nums = len(self.item_list)
        
        self.sample_start = []
        
        self.total_len = 0
        if self.opt.isTrain:
            cand_names = os.listdir(self.data_root_cand)
            self.img_candidates = []
            self.th_img_candidates = []
            for (idx, name) in enumerate(cand_names):
                output = imread(os.path.join(os.path.join(self.data_root_cand, name))).astype(np.float32)/255.
                output = cv2.resize(output, (512, 512))
                img_t = (output - 0.5) * 2  
                img_t = torch.from_numpy(np.transpose(img_t, axes=(2, 0 ,1)))
                self.img_candidates.append(output)
                self.th_img_candidates.append(img_t)
                if idx >= 3: break
        
            self.th_img_candidates = torch.cat(self.th_img_candidates)
            self.total_len = self.frame_nums
        else:
            # test mode        
            pass
    
    def _data_aug(self, img, l, r, u, b, h, w, a):
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), a, 1.0)
        img_rot = cv2.warpAffine(img, M, (w, h))

        t_img = img_rot[u:b, l:r]
        t_img = cv2.resize(t_img, (w, h))
        return t_img

    def positional_encoding(self, tensor, num_encoding_functions=6, include_input=True, log_sampling=True):
        """Apply positional encoding to the input."""
        # Trivially, the input tensor is added to the positional encoding.
        encoding = [tensor] if include_input else []
        frequency_bands = None
        if log_sampling:
            frequency_bands = 2.0**torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            frequency_bands = torch.linspace(
                2.0**0.0,
                2.0**(num_encoding_functions - 1),
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=0)
    
    def __getitem__(self, ind):
        data_index = ind

        # DA parameter
        h, w  = 512, 512
        if self.use_da:
            l = np.random.randint(0, 25)        # left
            r = w - l                           # right
            u = np.random.randint(0, 25)        # up
            b = h - u                           # bottom
            a = np.random.rand() * 30 - 15      # angle
        else:
            l, r, b, a = 0, w, h, 0
        
        tgt_image = np.asarray(Image.open(os.path.join(self.data_root_tar, self.item_list[data_index] + '.png')))
        tgt_image = cv2.resize(tgt_image, (512, 512)).astype(np.float32)/255.
        tgt_image = self._data_aug(tgt_image, l, r, u, b, h, w, a)
        tgt_image = (tgt_image - 0.5) * 2 
        tgt_image = torch.from_numpy(np.transpose(tgt_image, axes=(2, 0 ,1)))
                 
        feature_map = np.asarray(Image.open(os.path.join(self.data_root_feat, self.item_list[data_index] + '.png'))).astype(np.float32)/255.
        feature_map = self._data_aug(feature_map, l, r, u, b, h, w, a)
        feature_map = (feature_map - 0.5) * 2   
        feature_map = torch.from_numpy(np.transpose(feature_map, axes=(2, 0 ,1)))
        
        ## facial weight mask
        weight_mask = np.asarray(Image.open(os.path.join(self.data_root_weight, self.item_list[data_index] + '.png'))).astype(np.float32)/255.
        weight_mask = self._data_aug(weight_mask, l, r, u, b, h, w, a)
        weight_mask = (weight_mask - 0.5) * 2  
        weight_mask = torch.from_numpy(weight_mask).unsqueeze(0)

        t = ind / self.tpos_param  # hyper param
        t_tens = np.ones((h, w), np.float32) + t
        t_tens = self._data_aug(t_tens, l, r, u, b, h, w, a)
        t_embd = self.positional_encoding(torch.from_numpy(t_tens[None, ...])).float()
        # print('t_embd.shape:', t_embd.shape)
        
        return_list = {'feature_map': feature_map, 'cand_image': self.th_img_candidates, 't_embd': t_embd, 'tgt_image': tgt_image, 'weight_mask': weight_mask}
           
        return return_list

    def get_data_test_mode(self, landmarks, shoulder, pad=None):
        ''' get transformed data
        '''
       
        feature_map = torch.from_numpy(self.get_feature_image(landmarks, (self.opt.loadSize, self.opt.loadSize), shoulder, pad)[np.newaxis, :].astype(np.float32)/255.)

        return feature_map  
    
    def get_feature_image(self, landmarks, size, shoulders=None, image_pad=None):
        # draw edges
        im_edges = self.draw_face_feature_maps(landmarks, size)  
        if shoulders is not None:
            if image_pad is not None:
                top, bottom, left, right = image_pad
                delta_y = top - bottom
                delta_x = right - left
                shoulders[:, 0] += delta_x
                shoulders[:, 1] += delta_y
            im_edges = self.draw_shoulder_points(im_edges, shoulders)
        
        return im_edges

    def draw_shoulder_points(self, img, shoulder_points):
        # import pdb; pdb.set_trace()
        if shoulder_points.shape[0] >= 4:
            num = int(shoulder_points.shape[0] / 2)
            for i in range(2):
                for j in range(num - 1):
                    pt1 = [int(flt) for flt in shoulder_points[i * num + j]]
                    pt2 = [int(flt) for flt in shoulder_points[i * num + j + 1]]
                    img = cv2.line(img, tuple(pt1)[:2], tuple(pt2)[:2], (225, 225, 225), 2)  # BGR
        else:
            for j in range(shoulder_points.shape[0] - 1):
                pt1 = [int(flt) for flt in shoulder_points[j]]
                pt2 = [int(flt) for flt in shoulder_points[j + 1]]
                img = cv2.line(img, tuple(pt1)[:2], tuple(pt2)[:2], (225, 225, 225), 2)  # BGR
        
            for i in range(shoulder_points.shape[0]):
                img = cv2.circle(img, np.int32(shoulder_points[i])[:2], 8, (225, 225, 225), 1)
        
        return img
    
    def draw_face_feature_maps(self, keypoints, size=(512, 512)):
        w, h = size
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
        for edge_list in self.part_list:
            for edge in edge_list:
                for i in range(len(edge)-1):
                    pt1 = [int(flt) for flt in keypoints[edge[i]]]
                    pt2 = [int(flt) for flt in keypoints[edge[i + 1]]]
                    im_edges = cv2.line(im_edges, tuple(pt1), tuple(pt2), 255, 2)

        return im_edges
    
    def draw_landmarks(self, image, landmarks, connections, landmark_drawing_spec, connection_drawing_spec):
        t_landmarks = np.int32(landmarks[:, :2])
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            drawing_spec = connection_drawing_spec[connection] if isinstance(
                connection_drawing_spec, Mapping) else connection_drawing_spec
            cv2.line(image, t_landmarks[start_idx],
                    t_landmarks[end_idx], drawing_spec.color,
                    drawing_spec.thickness)
        
        if landmark_drawing_spec:
            for idx in range(t_landmarks.shape[0]):
                landmark_px = t_landmarks[idx]
                drawing_spec = landmark_drawing_spec[idx] if isinstance(
                landmark_drawing_spec, Mapping) else landmark_drawing_spec
            # White circle border
            circle_border_radius = max(drawing_spec.circle_radius + 1,
                                        int(drawing_spec.circle_radius * 1.2))
            cv2.circle(image, landmark_px, circle_border_radius, (224, 224, 224),
                        drawing_spec.thickness)
            # Fill color into the circle
            cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                        drawing_spec.color, drawing_spec.thickness)

    def semantic_meshs2figure(self, rlt, canvas):
        ret_img = canvas.copy()
        # draw FACEMESH_TESSELATION
        self.draw_landmarks(ret_img, rlt, 
            mp.solutions.face_mesh.FACEMESH_TESSELATION, 
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(), 
            landmark_drawing_spec=None)

        # draw FACEMESH_CONTOURS
        self.draw_landmarks(ret_img, rlt, 
            mp.solutions.face_mesh.FACEMESH_CONTOURS, 
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(), 
            landmark_drawing_spec=None)

        # draw FACEMESH_IRISES
        self.draw_landmarks(ret_img, rlt, 
            mp.solutions.face_mesh.FACEMESH_IRISES, 
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style(), 
            landmark_drawing_spec=None)
        
        return ret_img

    def get_semantic_indices(self):

        semantic_connections = {
            'Contours':     self.mp_connections.FACEMESH_CONTOURS,
            'FaceOval':     self.mp_connections.FACEMESH_FACE_OVAL,
            'LeftIris':     self.mp_connections.FACEMESH_LEFT_IRIS,
            'LeftEye':      self.mp_connections.FACEMESH_LEFT_EYE,
            'LeftEyebrow':  self.mp_connections.FACEMESH_LEFT_EYEBROW,
            'RightIris':    self.mp_connections.FACEMESH_RIGHT_IRIS,
            'RightEye':     self.mp_connections.FACEMESH_RIGHT_EYE,
            'RightEyebrow': self.mp_connections.FACEMESH_RIGHT_EYEBROW,
            'Lips':         self.mp_connections.FACEMESH_LIPS,
            'Tesselation':  self.mp_connections.FACEMESH_TESSELATION
        }

        def get_compact_idx(connections):
            ret = []
            for conn in connections:
                ret.append(conn[0])
                ret.append(conn[1])
            
            return sorted(tuple(set(ret)))
        
        semantic_indexes = {k: get_compact_idx(v) for k, v in semantic_connections.items()}

        return semantic_indexes

    def generate_facial_weight_mask(self, points, img_shape=(512, 512), mouth_outer=None):
        mouth_mask = np.zeros([*img_shape, 1])
        points = points[mouth_outer]
        points = np.int32(points[..., :2])
        mouth_mask = cv2.fillPoly(mouth_mask, [points], (255,0,0))
        mouth_mask = cv2.dilate(mouth_mask, np.ones((45, 45)))
        
        return mouth_mask

    def get_feature_image(self, kps, shoulder_pts, image_shape=(512, 512)):
        canvas = np.zeros((*image_shape, 3), dtype=np.uint8) + 10
        im_edges = self.semantic_meshs2figure(np.int32(kps), canvas)  
        im_edges = self.draw_shoulder_points(im_edges, np.int32(shoulder_pts))
        mouth_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        im_mouth = self.generate_facial_weight_mask(kps, image_shape, mouth_outer)

        return im_edges, im_mouth

    def __len__(self):  
        return self.total_len

    def name(self):
        return 'RenderDataset'
    

