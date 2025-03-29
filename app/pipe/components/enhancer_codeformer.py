import os
import cv2
import torch
import numpy as np
from ray import serve
from torchvision.transforms.functional import normalize
from app.utils.utils import conditional_download, Face

# Add CodeFormer's path to sys.path to ensure imports work
import sys
codeformer_path = os.path.join(os.path.dirname(__file__), 'codeformer')
if codeformer_path not in sys.path:
    sys.path.append(codeformer_path)

# Now import from CodeFormer
from app.pipe.components.codeformer.basicsr.utils import img2tensor, tensor2img
from app.pipe.components.codeformer.basicsr.utils.misc import get_device
from app.pipe.components.codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from app.pipe.components.codeformer.basicsr.utils.registry import ARCH_REGISTRY

@serve.deployment()
class CodeFormerEnhancer:
    """
    A class used to enhance faces in a given frame using CodeFormer model.
    """

    def __init__(self):
        """
        Initializes the CodeFormerEnhancer class by downloading and setting up the models.
        """
        self.device = get_device()
        
        # Download detection and parsing models if needed
        conditional_download(
            'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
            'models/weights/facelib'
        )
        conditional_download(
            'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
            'models/weights/facelib'
        )
        
        # Load CodeFormer model
        self.net = ARCH_REGISTRY.get('CodeFormer')(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
            connect_list=['32', '64', '128', '256']
        ).to(self.device)
        
        ckpt_path = conditional_download(
            'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
            'models/weights/CodeFormer'
        )
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')['params_ema']
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        
        # Set up background upsampler (RealESRGAN)
        self.bg_upsampler = self._set_realesrgan()
        
        print(f'CodeFormerEnhancer is using device: {self.device}')

    def _set_realesrgan(self):
        """
        Set up the RealESRGAN model for background enhancement.
        
        Returns:
            RealESRGANer or None: The upsampler instance or None if not available
        """
        from app.pipe.components.codeformer.basicsr.archs.rrdbnet_arch import RRDBNet
        from app.pipe.components.codeformer.basicsr.utils.realesrgan_utils import RealESRGANer
        
        use_half = False
        if torch.cuda.is_available():  # set False in CPU/MPS mode
            no_half_gpu_list = ['1650', '1660']  # set False for GPUs that don't support f16
            if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
                use_half = True
        
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        
        upsampler = RealESRGANer(
            scale=2,
            model_path=conditional_download(
                'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth',
                'models/weights/realesrgan'
            ),
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=use_half
        )
        
        return upsampler

    def enhance_face(self, target_face, temp_frame, fidelity_weight=0.5, background_enhance=True, face_upsample=True, upscale=2):
        """
        Enhances the given face in the frame using CodeFormer and returns the enhanced frame.
        
        Args:
            target_face (Face): The target face namedtuple containing keypoints and embedding.
            temp_frame (np.ndarray): The input image frame.
            fidelity_weight (float): Balance the quality and fidelity (0 for better quality, 1 for better identity).
            background_enhance (bool): Whether to enhance the background.
            face_upsample (bool): Whether to upsample the face.
            upscale (int): The upscaling factor.
            
        Returns:
            np.ndarray: The enhanced frame.
        """
        if target_face is None:
            return None

        # Create face restoration helper
        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device
        )
        
        # Face helper settings
        face_helper.clean_all()
        face_helper.read_image(temp_frame)
        
        # Use the provided face landmarks instead of detecting
        face_helper.all_landmarks_5 = [target_face.kps]
        face_helper.align_warp_face()
        
        # Choose appropriate upsampler
        bg_upsampler = self.bg_upsampler if background_enhance else None
        face_upsampler = self.bg_upsampler if face_upsample else None
        
        # Process each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
            
            try:
                with torch.no_grad():
                    output = self.net(cropped_face_t, w=fidelity_weight, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f"\tFailed inference for CodeFormer: {error}")
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
                
            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)
        
        # Paste back
        face_helper.get_inverse_affine(None)
        
        # Upsample the background if needed
        if bg_upsampler is not None:
            bg_img = bg_upsampler.enhance(temp_frame, outscale=upscale)[0]
        else:
            bg_img = None
            
        # Paste each restored face to the input image
        if face_upsample and face_upsampler is not None:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img, draw_box=False, face_upsampler=face_upsampler
            )
        else:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img, draw_box=False
            )
            
        return restored_img


# For testing the component independently
if __name__ == '__main__':
    import os
    import sys
    import argparse
    import numpy as np
    from app.pipe.components.codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
    
    # Add paths for testing
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from app.utils.utils import Face
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Input image path')
    parser.add_argument('-o', '--output_path', type=str, default='output.jpg', help='Output image path')
    parser.add_argument('-w', '--fidelity_weight', type=float, default=0.5, help='Fidelity weight (0-1)')
    parser.add_argument('-bg', '--background_enhance', action='store_true', help='Enhance background')
    parser.add_argument('-fu', '--face_upsample', action='store_true', help='Upsample face')
    parser.add_argument('-s', '--upscale', type=int, default=2, help='Upscale factor')
    args = parser.parse_args()
    
    # Load the image
    img = cv2.imread(args.input_path)
    if img is None:
        print(f"Error: Cannot load image from {args.input_path}")
        sys.exit(1)
    
    # Create temporary face helper for detection only
    print("Detecting face...")
    face_helper = FaceRestoreHelper(
        1,  # Temporary upscale factor
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=get_device()
    )
    
    face_helper.read_image(img)
    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=False, resize=640, eye_dist_threshold=5
    )
    
    if num_det_faces == 0:
        print("Error: No face detected in the image")
        sys.exit(1)
    
    print(f"Detected {num_det_faces} face(s)")
    
    # Create a Face namedtuple from the first detected face
    kps = face_helper.all_landmarks_5[0]
    # Create a dummy embedding (since CodeFormerEnhancer only uses the keypoints)
    dummy_embedding = np.zeros(512, dtype=np.float32)
    face = Face(kps=kps, embedding=dummy_embedding)
    
    # Initialize CodeFormerEnhancer
    print("Initializing CodeFormerEnhancer...")
    codeformer = CodeFormerEnhancer()
    
    # Enhance face using the CodeFormerEnhancer class
    print(f"Enhancing face with fidelity_weight={args.fidelity_weight}...")
    enhanced_img = codeformer.enhance_face(
        face, img, 
        fidelity_weight=args.fidelity_weight,
        background_enhance=args.background_enhance,
        face_upsample=args.face_upsample,
        upscale=args.upscale
    )
    
    # Save the result
    cv2.imwrite(args.output_path, enhanced_img)
    print(f"Enhanced image saved to {args.output_path}")
    

#python -m app.pipe.components.enhancer_codeformer -i calvo.jpg -o calvo2.jpg -w 0.5 -fu -s 2