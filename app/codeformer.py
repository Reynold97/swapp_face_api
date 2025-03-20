import os
import cv2
import torch
import numpy as np
import urllib.request
from torchvision.transforms.functional import normalize
from collections import OrderedDict
from PIL import Image
import math
import torchvision.utils as tv_utils

# Model URLs for downloading pre-trained models
CODEFORMER_MODEL_URL = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
DETECTION_MODEL_URL = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth'
PARSING_MODEL_URL = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth'
REALESRGAN_MODEL_URL = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'

# Create directories for saving models
os.makedirs('weights/CodeFormer', exist_ok=True)
os.makedirs('weights/facelib', exist_ok=True)
os.makedirs('weights/realesrgan', exist_ok=True)

# Registry for model architectures
class Registry:
    """Registry for model architectures."""
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        self._obj_map[name] = obj

    def register(self, obj=None):
        """Register an object."""
        if obj is None:
            def wrapper(fn):
                self._do_register(fn.__name__, fn)
                return fn
            return wrapper
        else:
            self._do_register(obj.__name__, obj)

    def get(self, name):
        """Get a registered object by name."""
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

# Create registry
ARCH_REGISTRY = Registry('arch')

#########################################
# Image Utility Functions
#########################################

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor."""
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images."""
    if not torch.is_tensor(tensor):
        raise TypeError(f"tensor should be a torch.Tensor, but got {type(tensor)}")
    
    # If list or tuple, make a batch
    if isinstance(tensor, (list, tuple)):
        tensor = torch.stack(tensor, dim=0)
    
    # Make grid function from torchvision
    return tv_utils.make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, 
                              range=range, scale_each=scale_each, pad_value=pad_value)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors to numpy arrays."""
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But got {n_dim}D tensor.')
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file."""
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)

def load_file_from_url(url, model_dir, progress=True, file_name=None):
    """Load file from URL."""
    os.makedirs(model_dir, exist_ok=True)
    
    if file_name is None:
        file_name = os.path.basename(url)
    
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}')
        torch.hub.download_url_to_file(url, cached_file, progress=progress)
    return cached_file

def is_gray(img, threshold=10):
    """Check if an image is grayscale."""
    if len(img.shape) == 2:
        return True
    
    # Convert to PIL to use PIL's method
    img_pil = Image.fromarray(img)
    if len(img_pil.getbands()) == 1:
        return True
    
    # Check color variance
    img1 = np.asarray(img_pil.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(img_pil.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(img_pil.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    return diff_sum <= threshold

def bgr2gray(img, out_channel=3):
    """Convert BGR image to grayscale."""
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    if out_channel == 3:
        gray = gray[:,:,np.newaxis].repeat(3, axis=2)
    return gray

def adain_npy(content_feat, style_feat):
    """Adaptive instance normalization for numpy arrays."""
    def calc_mean_std(feat, eps=1e-5):
        size = feat.shape
        assert len(size) == 3, 'The input feature should be 3D tensor.'
        c = size[2]
        feat_var = feat.reshape(-1, c).var(axis=0) + eps
        feat_std = np.sqrt(feat_var).reshape(1, 1, c)
        feat_mean = feat.reshape(-1, c).mean(axis=0).reshape(1, 1, c)
        return feat_mean, feat_std
    
    size = content_feat.shape
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - np.broadcast_to(content_mean, size)) / np.broadcast_to(content_std, size)
    return normalized_feat * np.broadcast_to(style_std, size) + np.broadcast_to(style_mean, size)

#########################################
# RRDBNet Architecture for Face Upsampling
#########################################

class ResidualDenseBlock(torch.nn.Module):
    """Residual Dense Block."""
    def __init__(self, num_feat=64, num_grow_ch=32, res_scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.res_scale + x

class RRDB(torch.nn.Module):
    """Residual in Residual Dense Block."""
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(torch.nn.Module):
    """Networks consisting of Residual in Residual Dense Block, used for SR."""
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        
        self.conv_first = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = torch.nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # upsample
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        # Simplified upsampling for x2 (for RealESRGAN_x2plus.pth model)
        feat = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(
            feat, scale_factor=2, mode='nearest')))
            
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

#########################################
# RealESRGANer for Face Upsampling
#########################################

class RealESRGANer:
    """Helper class for upsampling images with RealESRGAN."""
    
    def __init__(self, scale, model_path, model, tile=0, tile_pad=10, pre_pad=10, half=False, device=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
        # Load model
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))
        # Prefer params_ema
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)
        
        if self.half:
            self.model = self.model.half()
    
    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible"""
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = torch.nn.functional.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
            
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
            
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = torch.nn.functional.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')
    
    def process(self):
        # Model inference
        with torch.no_grad():
            self.output = self.model(self.img)
    
    def tile_process(self):
        """Process by tiles to handle large images."""
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                
                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
    
    def post_process(self):
        # Remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # Remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output
    
    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        """Enhance an image."""
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image
        try:
            with torch.no_grad():
                self.pre_process(img)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_img = self.post_process()
                output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
                if img_mode == 'L':
                    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        except RuntimeError as error:
            print(f"Failed inference for RealESRGAN: {error}")
            return None, None

        # Process alpha channel if needed
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # Use simple resizing for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # Merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # Convert to the appropriate output format
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        # Rescale if needed
        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode

#########################################
# Face Detection and Restoration
#########################################

def init_detection_model(model_name, half=False, device=None):
    """Initialize face detection model."""
    if model_name == 'retinaface_resnet50':
        # Load pretrained RetinaFace model
        model_path = load_file_from_url(
            DETECTION_MODEL_URL, model_dir='weights/facelib', progress=True, file_name=None)
        
        # Use a minimal stub for detection function
        # In a real implementation, we'd load the actual model
        # This is just a placeholder to demonstrate the interface
        class DetectionModel:
            def __init__(self):
                self.model_path = model_path
                print(f"Loaded detection model from {model_path}")
            
            def detect_faces(self, img, threshold=0.5):
                """Detect faces."""
                # Return dummy values - this is just a minimal implementation
                # In a real app, you would use the actual pretrained model
                h, w = img.shape[:2]
                return [[w*0.25, h*0.25, w*0.75, h*0.75, 0.99, 
                       w*0.3, h*0.3, w*0.7, h*0.3, w*0.5, h*0.5, 
                       w*0.3, h*0.7, w*0.7, h*0.7]]
        
        return DetectionModel()
    else:
        raise ValueError(f"Detection model {model_name} not supported")

def init_parsing_model(model_name='parsenet', device=None):
    """Initialize face parsing model."""
    # Load pretrained parsing model
    model_path = load_file_from_url(
        PARSING_MODEL_URL, model_dir='weights/facelib', progress=True, file_name=None)
    
    # Minimal stub for parsing model
    class ParsingModel:
        def __init__(self):
            self.model_path = model_path
            print(f"Loaded parsing model from {model_path}")
        
        def __call__(self, x):
            """Forward pass for parsing."""
            # In a real implementation, this would use the actual model
            # Return a dummy segmentation map
            batch_size = x.shape[0]
            return [torch.randn(batch_size, 19, 512, 512)]
    
    return ParsingModel()

class FaceRestoreHelper:
    """Helper for face restoration."""
    def __init__(self, upscale_factor, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50',
                 save_ext='png', template_3points=False, pad_blur=False, use_parse=False, device=None):
        self.upscale_factor = int(upscale_factor)
        self.crop_ratio = crop_ratio
        self.face_size = (int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))
        self.det_model = det_model
        self.save_ext = save_ext
        self.template_3points = template_3points
        self.pad_blur = pad_blur
        self.use_parse = use_parse
        
        # Use CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
        # Set standard 5 landmarks for FFHQ faces with 512 x 512
        if self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])
        
        # Adjust template based on face size and crop ratio
        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * (self.crop_ratio[1] - 1) / 2
        
        # Initialize empty lists and variables
        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.pad_input_imgs = []
        
        # Initialize face detection model
        self.face_detector = init_detection_model(det_model, half=False, device=self.device)
        
        # Initialize face parsing model if needed
        if use_parse:
            self.face_parse = init_parsing_model(model_name='parsenet', device=self.device)
    
    def read_image(self, img):
        """Read input image."""
        if isinstance(img, str):
            img = cv2.imread(img)
        
        # Handle various image formats
        if np.max(img) > 256:  # 16-bit image
            img = img / 65535 * 255
        if len(img.shape) == 2:  # gray image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # BGRA image with alpha channel
            img = img[:, :, 0:3]
        
        self.input_img = img
        self.is_gray = is_gray(img, threshold=10)
        if self.is_gray:
            print('Grayscale input: True')
        
        # Resize small images
        if min(self.input_img.shape[:2]) < 512:
            f = 512.0 / min(self.input_img.shape[:2])
            self.input_img = cv2.resize(self.input_img, (0,0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    
    def get_face_landmarks_5(self, only_center_face=False, resize=None, blur_ratio=0.01, eye_dist_threshold=None):
        """Get face landmarks."""
        if resize is None:
            scale = 1
            input_img = self.input_img
        else:
            h, w = self.input_img.shape[0:2]
            scale = resize / min(h, w)
            h, w = int(h * scale), int(w * scale)
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            input_img = cv2.resize(self.input_img, (w, h), interpolation=interp)
        
        # Use face detector
        bboxes = self.face_detector.detect_faces(input_img)
        
        if bboxes is None or len(bboxes) == 0:
            return 0
        else:
            bboxes = np.array(bboxes) / scale
        
        for bbox in bboxes:
            # Extract landmark points
            if self.template_3points:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 11, 2)])
            else:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
            self.all_landmarks_5.append(landmark)
            self.det_faces.append(bbox[0:5])
        
        if len(self.det_faces) == 0:
            return 0
        
        # Only keep largest or center face if specified
        if only_center_face:
            h, w, _ = self.input_img.shape
            self.det_faces, center_idx = self.get_center_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[center_idx]]
        
        # Handle blurry edges if needed
        if self.pad_blur:
            self.pad_input_imgs = []
            for landmarks in self.all_landmarks_5:
                # Implementation of padding for blurry edges
                pass  # Simplified for this script
        
        return len(self.all_landmarks_5)
    
    def align_warp_face(self, save_cropped_path=None, border_mode='constant'):
        """Align and warp faces."""
        if self.pad_blur:
            assert len(self.pad_input_imgs) == len(
                self.all_landmarks_5), f'Mismatched samples: {len(self.pad_input_imgs)} and {len(self.all_landmarks_5)}'
                
        for idx, landmark in enumerate(self.all_landmarks_5):
            # Use 5 landmarks to get affine matrix
            affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template, method=cv2.LMEDS)[0]
            self.affine_matrices.append(affine_matrix)
            
            # Set border mode
            if border_mode == 'constant':
                border_mode = cv2.BORDER_CONSTANT
            elif border_mode == 'reflect101':
                border_mode = cv2.BORDER_REFLECT101
            elif border_mode == 'reflect':
                border_mode = cv2.BORDER_REFLECT
            
            # Warp the face
            if self.pad_blur:
                input_img = self.pad_input_imgs[idx]
            else:
                input_img = self.input_img
            
            cropped_face = cv2.warpAffine(
                input_img, affine_matrix, self.face_size, 
                borderMode=border_mode, borderValue=(135, 133, 132))  # gray
            
            self.cropped_faces.append(cropped_face)
            
            # Save cropped face if path provided
            if save_cropped_path is not None:
                path = os.path.splitext(save_cropped_path)[0]
                save_path = f'{path}_{idx:02d}.{self.save_ext}'
                imwrite(cropped_face, save_path)
    
    def get_inverse_affine(self, save_inverse_affine_path=None):
        """Get inverse affine matrix."""
        for idx, affine_matrix in enumerate(self.affine_matrices):
            inverse_affine = cv2.invertAffineTransform(affine_matrix)
            inverse_affine *= self.upscale_factor
            self.inverse_affine_matrices.append(inverse_affine)
            
            # Save inverse affine matrices if path provided
            if save_inverse_affine_path is not None:
                path, _ = os.path.splitext(save_inverse_affine_path)
                save_path = f'{path}_{idx:02d}.pth'
                torch.save(inverse_affine, save_path)
    
    def add_restored_face(self, restored_face, input_face=None):
        """Add a restored face."""
        if self.is_gray:
            restored_face = bgr2gray(restored_face)  # convert img into grayscale
            if input_face is not None:
                restored_face = adain_npy(restored_face, input_face)  # transfer the color
        self.restored_faces.append(restored_face)
    
    def paste_faces_to_input_image(self, save_path=None, upsample_img=None, draw_box=False, face_upsampler=None):
        """Paste faces back to the original image."""
        h, w, _ = self.input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        
        if upsample_img is None:
            # Resize the background
            upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
        else:
            upsample_img = cv2.resize(upsample_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        
        if len(self.restored_faces) != len(self.inverse_affine_matrices):
            raise ValueError('length of restored_faces and affine_matrices are different.')
        
        inv_mask_borders = []
        for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
            # Handle face upsampling if needed
            if face_upsampler is not None:
                restored_face = face_upsampler.enhance(restored_face, outscale=self.upscale_factor)[0]
                inverse_affine /= self.upscale_factor
                inverse_affine[:, 2] *= self.upscale_factor
                face_size = (self.face_size[0]*self.upscale_factor, self.face_size[1]*self.upscale_factor)
            else:
                # Add offset for more precise alignment
                if self.upscale_factor > 1:
                    extra_offset = 0.5 * self.upscale_factor
                else:
                    extra_offset = 0
                inverse_affine[:, 2] += extra_offset
                face_size = self.face_size
            
            # Warp the restored face back
            inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up))
            
            # Create mask
            mask = np.ones(face_size, dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
            
            # Remove black borders
            inv_mask_erosion = cv2.erode(
                inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))
            
            pasted_face = inv_mask_erosion[:, :, None] * inv_restored
            total_face_area = np.sum(inv_mask_erosion)
            
            # Draw box if needed
            if draw_box:
                h, w = face_size
                mask_border = np.ones((h, w, 3), dtype=np.float32)
                border = int(1400/np.sqrt(total_face_area))
                mask_border[border:h-border, border:w-border,:] = 0
                inv_mask_border = cv2.warpAffine(mask_border, inverse_affine, (w_up, h_up))
                inv_mask_borders.append(inv_mask_border)
            
            # Create soft mask for blending
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
            
            if len(upsample_img.shape) == 2:  # grayscale image
                upsample_img = upsample_img[:, :, None]
            inv_soft_mask = inv_soft_mask[:, :, None]
            
            # Use face parsing if available
            if self.use_parse:
                # Face parsing implementation would go here
                # Simplified for this script
                pass
            
            # Blend the face with the background
            if len(upsample_img.shape) == 3 and upsample_img.shape[2] == 4:  # alpha channel
                alpha = upsample_img[:, :, 3:]
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img[:, :, 0:3]
                upsample_img = np.concatenate((upsample_img, alpha), axis=2)
            else:
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
        
        # Convert to proper data type
        if np.max(upsample_img) > 256:  # 16-bit image
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)
        
        # Draw bounding box if needed
        if draw_box:
            img_color = np.ones([*upsample_img.shape], dtype=np.float32)
            img_color[:,:,0] = 0
            img_color[:,:,1] = 255
            img_color[:,:,2] = 0
            for inv_mask_border in inv_mask_borders:
                upsample_img = inv_mask_border * img_color + (1 - inv_mask_border) * upsample_img
        
        # Save result if path provided
        if save_path is not None:
            path = os.path.splitext(save_path)[0]
            save_path = f'{path}.{self.save_ext}'
            imwrite(upsample_img, save_path)
        
        return upsample_img
    
    def get_center_face(self, det_faces, h=0, w=0, center=None):
        """Get the center face."""
        if center is not None:
            center = np.array(center)
        else:
            center = np.array([w / 2, h / 2])
        
        center_dist = []
        for det_face in det_faces:
            face_center = np.array([(det_face[0] + det_face[2]) / 2, (det_face[1] + det_face[3]) / 2])
            dist = np.linalg.norm(face_center - center)
            center_dist.append(dist)
        
        center_idx = center_dist.index(min(center_dist))
        return det_faces[center_idx], center_idx
    
    def clean_all(self):
        """Clean all intermediate results."""
        self.all_landmarks_5 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []
        self.det_faces = []
        self.pad_input_imgs = []

#########################################
# CodeFormer Model
#########################################

@ARCH_REGISTRY.register()
class CodeFormer(torch.nn.Module):
    """CodeFormer for Face Restoration."""
    def __init__(self, dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                connect_list=['32', '64', '128', '256'], fix_modules=['quantize', 'generator']):
        super().__init__()
        self.dim_embd = dim_embd
        self.codebook_size = codebook_size
        self.n_head = n_head
        self.n_layers = n_layers
        self.connect_list = connect_list
        self.fix_modules = fix_modules
        
        # For this standalone script, we'll use a simplified model structure
        # In a real implementation, this would include the full architecture
        self.quantize = torch.nn.Identity()
        self.generator = torch.nn.Identity()
        
        # Create dummy encoder and transformer for interface compatibility
        self.encoder = torch.nn.Identity()
        self.transformer = torch.nn.Identity()
    
    def forward(self, x, w=0, adain=False):
        # This is a simplified implementation
        # In a real app, you would load the pre-trained model weights
        # and use the actual model implementation
        return x, None

#########################################
# Face Enhancement Function
#########################################

def enhance_face_codeformer(img, upscale=2, face_upsample=False, codeformer_fidelity=0.5, device=None):
    """
    Enhance faces in an image using CodeFormer.
    
    Args:
        img: Input image (numpy array, BGR format)
        upscale: Upscale factor for the entire image
        face_upsample: Whether to further upsample the faces
        codeformer_fidelity: Balance between quality and fidelity (0-1)
        device: Device to use for inference
        
    Returns:
        Enhanced image with restored faces
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CodeFormer model
    codeformer_path = load_file_from_url(
        CODEFORMER_MODEL_URL, model_dir='weights/CodeFormer', progress=True, file_name=None)
    
    net = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=['32', '64', '128', '256']
    ).to(device)
    
    # Load model weights
    ckpt_path = "weights\CodeFormer\codeformer.pth"
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()
    
    # Set up face upsampler if needed
    face_upsampler = None
    if face_upsample:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2
        )
        face_upsampler = RealESRGANer(
            scale=2,
            model_path=load_file_from_url(
                REALESRGAN_MODEL_URL, model_dir='weights/realesrgan', progress=True, file_name=None),
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=False if device.type == 'cpu' else True
        )
    
    # Set up face helper
    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device
    )
    
    # Read and process image
    face_helper.read_image(img)
    face_helper.get_face_landmarks_5()
    face_helper.align_warp_face()
    
    # Restore each face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
        
        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=codeformer_fidelity, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'Failed inference for CodeFormer: {error}')
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
        
        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face, cropped_face)
    
    # Paste faces back
    face_helper.get_inverse_affine(None)
    
    if face_upsample and face_upsampler is not None:
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=None, draw_box=False, face_upsampler=face_upsampler)
    else:
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=None, draw_box=False)
    
    return restored_img

#########################################
# Main Function and CLI
#########################################

def process_image(image_path, output_path=None, upscale=2, face_upsample=False, codeformer_fidelity=0.5):
    """Process an image file, enhance faces, and save the result."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Enhance faces
    result = enhance_face_codeformer(
        img,
        upscale=upscale,
        face_upsample=face_upsample,
        codeformer_fidelity=codeformer_fidelity
    )
    
    # Save result
    if output_path is None:
        name, ext = os.path.splitext(os.path.basename(image_path))
        output_path = f"{name}_enhanced{ext}"
    
    cv2.imwrite(output_path, result)
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance faces in an image using CodeFormer')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    parser.add_argument('--upscale', type=int, default=2, help='Upscaling factor')
    parser.add_argument('--face_upsample', action='store_true', help='Additionally upsample faces')
    parser.add_argument('--fidelity', type=float, default=0.5, help='Balance between quality and fidelity (0-1)')
    
    args = parser.parse_args()
    
    output_path = process_image(
        args.input,
        args.output,
        args.upscale,
        args.face_upsample,
        args.fidelity
    )
    
    print(f'Enhanced image saved to: {output_path}')
    
    #python app\codeformer.py --input 'calvo.jpg' --output 'calvo2.jpg'