import os
import sys
import cv2
import torch
import pickle
import torchvision
import shutil
import glob
from tqdm import tqdm

def find_nlf_model():
    """Find NLF model file in root directory or models directory"""
    # Check for model files in root directory
    root_patterns = ['nlf_l_multi.torchscript', 'nlf_l_multi_*.torchscript']
    for pattern in root_patterns:
        matches = glob.glob(pattern)
        if matches:
            print(f"Found NLF model: {matches[0]}")
            return matches[0]
    
    # Check in models directory
    models_patterns = ['models/nlf_l_multi.torchscript', 'models/nlf_l_multi_*.torchscript']
    for pattern in models_patterns:
        matches = glob.glob(pattern)
        if matches:
            # Move to root directory
            model_path = matches[0]
            root_path = os.path.basename(model_path)
            print(f"Found NLF model in models directory: {model_path}")
            print(f"Moving to root directory as: {root_path}")
            shutil.move(model_path, root_path)
            return root_path
    
    return None

def download_nlf_model():
    """Download NLF model if not found locally"""
    print("NLF model not found locally.")
    print("\nTo use this script, you need the NLF model file.")
    print("Please download the model from the official source and place it in the root directory.")
    print("Expected filename: nlf_l_multi.torchscript or nlf_l_multi_<version>.torchscript")
    print("\nAlternatively, if you have the model in the 'models/' directory, it will be automatically copied.")
    sys.exit(1)

# Find or download the model
model_path = find_nlf_model()
if model_path is None:
    download_nlf_model()

# Load the pre-trained model
print(f"Loading NLF model from: {model_path}")
model = torch.jit.load(model_path).cuda().eval()

def process_video(video_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + '.pkl')
    if os.path.exists(output_file):
        print(f"file exists: {output_file}")
        return
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_results = {
        'joints3d_nonparam': [],
    }

    with torch.inference_mode(), torch.device('cuda'):
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to tensor
            frame_tensor = torch.from_numpy(frame).cuda()
            frame_batch = frame_tensor.unsqueeze(0).permute(0,3,1,2)
            # Model inference
            pred = model.detect_smpl_batched(frame_batch)
            # Collect pose data
            for key in pose_results.keys():
                if key in pred:
                    #pose_results[key].append(pred[key].cpu().numpy())
                    pose_results[key].append(pred[key])
                else:
                    pose_results[key].append(None)

            frame_idx += 1

    cap.release()

    # Prepare output data
    output_data = {
        'video_path': video_path,
        'video_length': frame_count,
        'video_width': video_width,
        'video_height': video_height,
        'pose': pose_results
    }

    # Save to pkl file
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)


def check_video_file(output_video_path):
    if os.path.exists(output_video_path):
        print(f"File {output_video_path} saved.")
        print(f"File size: {os.path.getsize(output_video_path) / 1024} KB")
    else:
        print(f"File {output_video_path} not found.")


def generate_video(image_paths, output_path):
    frame = cv2.imread(image_paths[0])
    if frame is None:
        return
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    if not video.isOpened():
        return

    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        video.write(frame)

    video.release()
    check_video_file(output_path)


def process_directory(video_dir):
    for root, _, files in os.walk(video_dir):
        has_video = False
        images = []
        for file in tqdm(files, desc="Processing files", unit="file"):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                output_dir = root + '_smpl'
                # print(output_dir)
                print(video_path)
                process_video(video_path, output_dir)



if __name__ == '__main__':
    video_directory = sys.argv[-1]
    process_directory(video_directory)
