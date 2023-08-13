import os
import time
import cv2
import wandb
import imageio
import subprocess
import numpy as np
from utils import util
import collections
import pandas as pd
import mediapipe as mp
from typing import Mapping
from datetime import datetime
from math import cos, sin, sqrt
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.win_size = opt.display_winsize
        self.name = opt.name
        if opt.isTrain:
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.task, opt.name, opt.backbone, 'logs')
            os.makedirs(self.log_dir, exist_ok=True)
            self.logger = wandb
            self.logger.init(project=opt.name + "-project", dir=self.log_dir, name=self.opt.task)
    
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            os.makedirs(os.path.dirname(self.log_name), exist_ok=True)
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        pass

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors):
        self.logger.log(errors)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, is_print=True):
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        message = '(epoch: %d, iters: %d, time: %s) ' % (epoch, i, dt_string)
        for k, v in sorted(errors.items()):
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        if is_print:
            print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
        return message
    
    def recover_image_with_hf(self, raw_data, win_size=15):
        image_base = (raw_data[..., :3] / 2 + 0.5) * 255
        image_hf = (raw_data[..., 3:] / 2 + 0.5) * 125.5
        h, w = image_base.shape[:2]

        ret_imgs = []
        for i in range(image_base.shape[2]):
            b_img = image_base[..., i]
            fb = np.fft.fft2(b_img)   
            fbshift = np.fft.fftshift(fb)

            h_img = image_hf[..., i]
            fh = np.fft.fft2(h_img)   
            fhshift = np.fft.fftshift(fh)

            crow, ccol = int(h/2),int(w/2)
            fbshift[crow-win_size: crow+win_size, ccol-win_size:ccol+win_size] = fhshift[crow-win_size: crow+win_size, ccol-win_size:ccol+win_size]

            ishift = np.fft.ifftshift(fbshift)
            iimg = np.fft.ifft2(ishift)
            iimg = np.abs(iimg)
            ret_imgs.append(iimg)

        return np.uint8(cv2.merge(ret_imgs))

    # save image to the disk
    def save_images(self, image_dir, visuals, image_path, webpage=None):        
        dirname = os.path.basename(os.path.dirname(image_path[0]))
        image_dir = os.path.join(image_dir, dirname)
        util.mkdir(image_dir)
        name = image_path
#        name = os.path.basename(image_path[0])
#        name = os.path.splitext(name)[0]        

        if webpage is not None:
            webpage.add_header(name)
            ims, txts, links = [], [], []         

        for label, t_image in visuals.items():
            save_ext = 'jpg'
            image_name = '%s_%s.%s' % (label, name, save_ext)
            save_path = os.path.join(image_dir, image_name)
            if len(t_image.shape) == 5:
                image_numpy = t_image.detach().cpu().numpy()[0, 0]
            else:
                image_numpy = t_image.detach().cpu().numpy()[0]
            image_numpy = np.transpose(image_numpy, axes=(1, 2, 0))

            # here
            # if image_numpy.shape[-1] == 6:
            #     # image_numpy = self.recover_image_with_hf(image_numpy)
            #     pass
            # else:
            image_numpy = np.uint8((image_numpy / 2 + 0.5) * 255)
            if image_numpy.shape[-1] > 1:
                image_numpy = image_numpy[..., :3]
            else:
                image_numpy = image_numpy[..., 0]
            util.save_image(image_numpy, save_path)

            if webpage is not None:
                ims.append(image_name)
                txts.append(label)
                links.append(image_name)
        if webpage is not None:
            webpage.add_images(ims, txts, links, width=self.win_size)

    def vis_print(self, message):
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
full_face_pred_types = {'face':     pred_type(slice(0, 17), (0.682, 0.780, 0.909, 1.0)),
                        'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.9)),
                        'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.8)),
                        'nose':     pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.9)),
                        'nostril':  pred_type(slice(31, 36), (0.545, 0.139, 0.643, 0.9)),
                        'eye1':     pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.9)),
                        'eye2':     pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.8)),
                        'lips':     pred_type(slice(48, 60), (0.870, 0.244, 0.210, 0.8)),
                        'teeth':    pred_type(slice(60, 68), (0.990, 0.975, 0.899, 0.9))
                        }
only_mouth_pred_types = {'lips':    pred_type(slice(48-48, 60-48), (0.870, 0.244, 0.210, 0.8)),
                         'teeth':   pred_type(slice(60-48, 68-48), (0.990, 0.975, 0.899, 0.9)),
                         'face':    pred_type(slice(20, 27),       (0.682, 0.780, 0.909, 1.0)),
                         }

full_face_pred_types_73 = {'mouth-1': pred_type(slice(4, 11), (0.870, 0.244, 0.210, 0.8)),
                           'mouth-2': pred_type(slice(46, 64), (1.0, 0.898, 0.755, 0.9)),
                           'eyebrow-1': pred_type(slice(27, 34), (0.596, 0.875, 0.541, 0.9)),
                           'eyebrow-2': pred_type(slice(65, 73), (0.596, 0.875, 0.541, 0.8)),
                           'o-face-1': pred_type(slice(0, 4), (1.0, 0.498, 0.055, 0.8)),
                           'o-face-2': pred_type(slice(11, 15), (1.0, 0.498, 0.055, 0.8)),
                           'o-eyebrows': pred_type(slice(15, 27), (0.345, 0.239, 0.443, 0.9)),
                           'o-nose': pred_type(slice(35, 46), (0.682, 0.780, 0.909, 1.0)),
                           }

only_mouth_pred_types_73 = {'mouth-1': pred_type(slice(4-4, 11-4), (0.870, 0.244, 0.210, 0.8)),
                            'mouth-2': pred_type(slice(46-46+7, 64-46+7), (1.0, 0.898, 0.755, 0.9)),
                            }


def format_cv_colors(x):
    """from float color to rgba color 0~255 for plotly

    Args:
        x (_type_): _description_
    """
    def _c(num):
        return int(num*255)
    return (_c(x[2]), _c(x[1]), _c(x[0]))


def landmarks2figure(landmarks, canvas=None, image_shape=(512, 512), with_line=True, only_mouth=False, is_kps73=False, norm_73=True):
    """
    landmarks (np.ndarray): 68 or 73 3D facial keypoints, shape: (68|73, 3)
    image_shape (tuple): output image shape. Defaults to (512, 512)
    """
    if canvas is None:
        canvas = np.uint8(np.zeros((*image_shape, 3)) + 183)
    else:
        image_shape = canvas.shape[:2]
    
    if not is_kps73:
        t_pred_types = only_mouth_pred_types if only_mouth else full_face_pred_types
    else:
        t_pred_types = only_mouth_pred_types_73 if only_mouth else full_face_pred_types_73
        if norm_73:
            # norm kps73
            landmarks[:, :2] = landmarks[:, :2] / 2 + 0.5
            landmarks[:, 1] = 1 - landmarks[:, 1]
            landmarks[:, 0] = landmarks[:, 0] * image_shape[1]
            landmarks[:, 1] = landmarks[:, 1] * image_shape[0]


    for k, v in t_pred_types.items():
        kps = landmarks[v.slice, :2]
        # draw line
        if with_line:
            for i in range(len(kps) - 1):
                canvas = cv2.line(canvas, (int(kps[i][0]), int(kps[i][1])), 
                                    (int(kps[i+1][0]), int(kps[i+1][1])), 
                                    format_cv_colors(v.color), 
                                    thickness=2, lineType=cv2.LINE_AA)
            # compensate eye
            if ('eye' in k and 'brow' not in k) or 'teeth' in k or 'lips' in k:
                canvas = cv2.line(canvas, (int(kps[0][0]), int(kps[0][1])), 
                                    (int(kps[-1][0]), int(kps[-1][1])), 
                                    format_cv_colors(v.color), 
                                    thickness=2, lineType=cv2.LINE_AA)
            # draw kps                  
            for i in range(len(kps)):
                canvas = cv2.circle(canvas, (int(kps[i][0]), int(kps[i][1])), 
                                    4, format_cv_colors(v.color), 1, lineType=cv2.LINE_AA)
        else:
            # draw kps
            for i in range(len(kps)):
                canvas = cv2.circle(canvas, (int(kps[i][0]), int(kps[i][1])), 
                                    2, format_cv_colors(v.color), -1, lineType=cv2.LINE_AA)
    
    return canvas


def recover_pts(headpose, trans_before, trans_after, pts3d, face_center=np.array((256, 256, 128))):
    t_pts = pts3d - face_center
    t_pts = t_pts + trans_after
    rx, ry, rz = headpose[0], headpose[1], headpose[2]
    rot = R.from_euler('xyz', [ry, rx, rz], degrees=True).inv()
    t_pts = rot.apply(t_pts)
    t_pts = t_pts + trans_before
    return t_pts


def denorm_face_kps3d(normed_kps3d, pose, t_fc_before, t_fc_after, face_center=np.array((256, 256, 128))):

    t_kps3d = normed_kps3d - np.stack([face_center]*normed_kps3d.shape[0])
    t_kps3d = t_kps3d + np.stack([t_fc_after]*t_kps3d.shape[0])

    rx, ry, rz = pose[0], pose[1], pose[2]
    rot_inv = R.from_euler('xyz', [ry, rx, rz], degrees=True).inv()

    kps3d_card = rot_inv.apply(t_kps3d)

    t_kps3d = kps3d_card + np.stack([t_fc_before]*t_kps3d.shape[0])

    return t_kps3d


def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def build_camera_box(rear_size=90):
    point_3d = []
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = int(4 / 3 * rear_size)
    front_depth = int(4 / 3 * rear_size)
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    return point_3d


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z


def angle2matrix(angles, gradient='false'):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
        gradient(str): whether to compute gradient matrix: dR/d_x,y,z
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x), -sin(x)],
                 [0, sin(x),  cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    
    R=Rz.dot(Ry.dot(Rx))
    #R=Rx.dot(Ry.dot(Rz))
    
    if gradient != 'true':
        return R.astype(np.float32)
    elif gradient == 'true':
        # gradident matrix
        dRxdx = np.array([[0,      0,       0],
                          [0, -sin(x), -cos(x)],
                          [0, cos(x),  -sin(x)]])
        dRdx = Rz.dot(Ry.dot(dRxdx)) * np.pi/180
        dRydy = np.array([[-sin(y), 0,  cos(y)],
                          [      0, 0,       0],
                          [-cos(y), 0, -sin(y)]])
        dRdy = Rz.dot(dRydy.dot(Rx)) * np.pi/180
        dRzdz = np.array([[-sin(z), -cos(z), 0],
                          [ cos(z), -sin(z), 0],
                          [     0,        0, 0]])
        dRdz = dRzdz.dot(Ry.dot(Rx)) * np.pi/180
        
        return R.astype(np.float32), [dRdx.astype(np.float32), dRdy.astype(np.float32), dRdz.astype(np.float32)]


def plot_pose_box(img, P, ver, color=(40, 255, 0), line_width=2):
    """ Draw a 3D box as annotation of pose.
    Ref:https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
    Args:
        img: the input image
        P: (3, 4). Affine Camera Matrix.
        kpt: (2, 68) or (3, 68)
    """
    llength = calc_hypotenuse(ver)
    point_3d = build_camera_box(llength)
    # Map to 2d image points
    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
    point_2d = point_3d_homo.dot(P.T)[:, :2]

    point_2d[:, 1] = - point_2d[:, 1]
    point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(ver[:27, :2], 0)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

    return img


def headpose_to_figure(headpose, pts3d, trans_before=None, trans_after=None, canvas=None, image_shape=(512, 512), face_center=np.array((256, 256, 128)), need_recover_pts=True, is_kps73=False, norm_73=True):
    if need_recover_pts:
        t_pts = denorm_face_kps3d(pts3d, headpose, trans_before, trans_after)
    else:
        t_pts = pts3d.copy()

    canvas = landmarks2figure(t_pts.copy(), image_shape=image_shape, canvas=canvas, is_kps73=is_kps73, norm_73=norm_73)

    P = np.hstack((angle2matrix(headpose), np.array([[6.6240250e+01, 6.2636879e+01, -6.6674835e+01]]).T))

    ret_img = plot_pose_box(canvas, P, t_pts)

    return ret_img


def frames_to_video(frames, save_pwd, fps=float(60), audio_fp=None):
    os.makedirs(os.path.dirname(save_pwd), exist_ok=True)
    video_tmp_path = save_pwd.replace('.mp4', 'tmp.mp4')
    writer = imageio.get_writer(video_tmp_path, fps=fps)

    for frame in frames:
        writer.append_data(frame[..., ::-1])
    
    writer.close()

    if audio_fp is not None and os.path.exists(audio_fp):
        cmd = 'ffmpeg -i "' + video_tmp_path + '" -i "' + audio_fp + '" -c:a aac -b:a 128k -shortest "' + save_pwd + '" -y'
        subprocess.call(cmd, shell=True) 
        os.remove(video_tmp_path)  # remove the template video


mp_drawing_styles = mp.solutions.drawing_styles
mp_connections = mp.solutions.face_mesh_connections

def draw_landmarks(image, landmarks, connections, landmark_drawing_spec, connection_drawing_spec):
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


def semantic_meshs2figure(rlt, canvas):
    ret_img = canvas.copy()
    # draw FACEMESH_TESSELATION
    draw_landmarks(ret_img, rlt, 
        mp.solutions.face_mesh.FACEMESH_TESSELATION, 
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(), 
        landmark_drawing_spec=None)

    # draw FACEMESH_CONTOURS
    draw_landmarks(ret_img, rlt, 
        mp.solutions.face_mesh.FACEMESH_CONTOURS, 
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(), 
        landmark_drawing_spec=None)

    # draw FACEMESH_IRISES
    draw_landmarks(ret_img, rlt, 
        mp.solutions.face_mesh.FACEMESH_IRISES, 
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(), 
        landmark_drawing_spec=None)
    
    return ret_img


def compute_lips3d_metric_headmotion(t_pred, t_gt):

    mse  = mean_squared_error(t_pred,  t_gt)
    rmse = mean_squared_error(t_pred,  t_gt, squared=False)
    abs  = mean_absolute_error(t_pred, t_gt)

    ret = {'MSE': mse, 'RMSE': rmse, 'L1': abs}

    return ret


def compute_lips3d_metric(t_pred, t_gt, semantic_indices=None, is_2d=True):

    valid_slice = slice(0, -1) if semantic_indices is None else semantic_indices
    dim_slice = slice(0, 2) if is_2d else slice(0, -1)

    mse  = mean_squared_error(t_pred[valid_slice, dim_slice],  t_gt[valid_slice, dim_slice])
    rmse = mean_squared_error(t_pred[valid_slice, dim_slice],  t_gt[valid_slice, dim_slice], squared=False)
    abs  = mean_absolute_error(t_pred[valid_slice, dim_slice], t_gt[valid_slice, dim_slice])

    ret = {'MSE': mse, 'RMSE': rmse, 'L1': abs}

    return ret


def results_to_csv(result_list, out_fp):
    meta = {k:[] for k in result_list[0].keys()}

    # os.makedirs(os.path.dirname(out_fp), exist_ok=True)

    for item in result_list:
        for k in meta.keys():
            meta[k].append(item[k])
    
    for k in meta.keys():
        if k == 'name' or isinstance(meta[k][0], str):
            meta[k].append('Average')
        else:
            meta[k].append(sum(meta[k])/len(meta[k]))

    df = pd.DataFrame(meta)
    df.to_csv(out_fp)


def load_txt(fp):
    with open(fp) as fid:
        datasets = [x.strip() for x in fid.readlines()]
        datasets = [x.split('\t') for x in datasets]

    return datasets


def get_semantic_indices():

    semantic_connections = {
        'Contours':     mp_connections.FACEMESH_CONTOURS,
        'FaceOval':     mp_connections.FACEMESH_FACE_OVAL,
        'LeftIris':     mp_connections.FACEMESH_LEFT_IRIS,
        'LeftEye':      mp_connections.FACEMESH_LEFT_EYE,
        'LeftEyebrow':  mp_connections.FACEMESH_LEFT_EYEBROW,
        'RightIris':    mp_connections.FACEMESH_RIGHT_IRIS,
        'RightEye':     mp_connections.FACEMESH_RIGHT_EYE,
        'RightEyebrow': mp_connections.FACEMESH_RIGHT_EYEBROW,
        'Lips':         mp_connections.FACEMESH_LIPS,
        'Tesselation':  mp_connections.FACEMESH_TESSELATION
    }

    def get_compact_idx(connections):
        ret = []
        for conn in connections:
            ret.append(conn[0])
            ret.append(conn[1])
        
        return sorted(tuple(set(ret)))
    
    semantic_indexes = {k: get_compact_idx(v) for k, v in semantic_connections.items()}

    return semantic_indexes


def denorm_verticies_via_headpose(vertices, hp, eye_center, scale, face_center=np.array((256, 235, 128)), from_3ddfa=True):
    vertices_canonical = (vertices - face_center) / scale

    rx, ry, rz = hp[0], hp[1], hp[2]
    if from_3ddfa:
        rot = R.from_euler('xyz', [ry, rx, rz], degrees=True).inv()
    else:
        rot = R.from_euler('xyz', [ry, -rx, -rz], degrees=True).inv()

    vertices_center = rot.apply(vertices_canonical)
    ret_ver = vertices_center + eye_center

    return ret_ver


def draw_shoulder_points(img, shoulder_points):
    num = int(shoulder_points.shape[0] / 2)
    for i in range(2):
        for j in range(num - 1):
            pt1 = [int(flt) for flt in shoulder_points[i * num + j]]
            pt2 = [int(flt) for flt in shoulder_points[i * num + j + 1]]
            img = cv2.line(img, tuple(pt1)[:2], tuple(pt2)[:2], (225, 225, 225), 2)  # BGR
    
    return img

    
def draw_face_feature_maps(keypoints, size=(512, 512), part_list=None):
    w, h = size
    # edge map for face region from keypoints
    im_edges = np.zeros((h, w, 3), np.uint8) # edge map for all edges
    for edge_list in part_list:
        for edge in edge_list:
            for i in range(len(edge)-1):
                pt1 = [int(flt) for flt in keypoints[edge[i]]]
                pt2 = [int(flt) for flt in keypoints[edge[i + 1]]]
                im_edges = cv2.line(im_edges, tuple(pt1)[:2], tuple(pt2)[:2], 255, 2)

    return im_edges


def draw_landmarks(image, landmarks, connections, landmark_drawing_spec, connection_drawing_spec):
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


def get_feature_image(landmarks, size=None, shoulders=None, image_pad=None, canvas=None):
    # draw edges
    if canvas is None:
        if size is not None:
            h, w = size
        else:
            h, w = 512, 512
        canvas = np.zeros((h, w, 3), np.uint8)
    if len(landmarks.shape) > 2:
        im_edges = landmarks
    else:
        im_edges = semantic_meshs2figure(landmarks, canvas.copy())  
    if shoulders is not None:
        if image_pad is not None:
            top, bottom, left, right = image_pad
            delta_y = top - bottom
            delta_x = right - left
            shoulders[:, 0] += delta_x
            shoulders[:, 1] += delta_y
        im_edges = draw_shoulder_points(im_edges, shoulders)
    
    return im_edges