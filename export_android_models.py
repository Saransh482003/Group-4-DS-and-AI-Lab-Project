import os
import sys
import torch

def export_yolo():
    print("--- Exporting YOLO ---")
    try:
        from ultralytics import YOLO
        yolo_path = "model_training/object_detection/best-weights/YOLO11s-Final-Training.pt"
        
        if not os.path.exists(yolo_path):
            print(f"[-] YOLO model not found at: {yolo_path}")
            return
        
        print(f"[+] Loading YOLO from {yolo_path}")
        model = YOLO(yolo_path)
            
        print("[+] Exporting YOLO to ONNX...")
        # Strictly enforce opset 14
        model.export(format='onnx', imgsz=640, opset=14, simplify=True)
        print("[+] YOLO Export completed.")
        
    except ImportError:
        print("[-] Ultralytics not installed. 'pip install ultralytics onnx tensorflow'")


def export_depth_anything():
    print("\n--- Exporting Depth-Anything-V2 ---")
    
    # We need the local repository architectures to load the .pth weights
    sys.path.append(os.path.abspath("Depth-Anything-V2/metric_depth"))
    
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        depth_path = "model_training/depth_estimation/model_weights/depth_anything_v2_metric_hypersim_vits.pth"
        
        if not os.path.exists(depth_path):
            print(f"[-] Depth model not found at: {depth_path}")
            return

        print(f"[+] Loading Depth-Anything-V2 from {depth_path}")
        
        # ViT-Small configuration (vits)
        model_configs = {
            'encoder': 'vits', 
            'features': 64, 
            'out_channels': [48, 96, 192, 384]
        }
        
        # Initialize metric depth model (Hypersim max_depth is typically evaluated at 20.0 or 80.0, we will use 20.0 for indoor edge case)
        depth_model = DepthAnythingV2(**model_configs, max_depth=20.0)
        
        # Load the PyTorch weights
        state_dict = torch.load(depth_path, map_location='cpu')
        
        # Depending on how it was saved, it might have 'model' key
        if 'model' in state_dict:
            state_dict = state_dict['model']
            
        depth_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        depth_model.eval()

        onnx_output_path = "model_training/depth_estimation/model_weights/depth_anything_v2_vits.onnx"
        
        # Standard input size for Depth-Anything V2 is 518x518
        dummy_input = torch.randn(1, 3, 518, 518)
        
        print(f"[+] Exporting Depth model to {onnx_output_path} (This might take a minute)...")
        torch.onnx.export(
            depth_model, 
            dummy_input, 
            onnx_output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 
                'output': {0: 'batch_size', 1: 'height', 2: 'width'}
            }
        )
        print("[+] Depth-Anything-V2 ONNX Export completed.")
        
    except ImportError as e:
        print(f"[-] Missing library or path issue for Depth-Anything: {e}")

if __name__ == "__main__":
    export_yolo()
    export_depth_anything()
    print("\n[+] Note: Piper TTS is already exported to ONNX format and located in 'piper/piper_voices/en_US-amy-medium.onnx'")
