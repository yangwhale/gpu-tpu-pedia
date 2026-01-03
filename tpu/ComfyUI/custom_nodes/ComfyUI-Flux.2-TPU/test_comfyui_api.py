#!/usr/bin/env python3
"""
通过 ComfyUI API 测试 Flux.2 TPU Pipeline
"""

import json
import requests
import time
import sys

COMFYUI_URL = "http://127.0.0.1:8189"

# Flux.2 TPU Full Pipeline workflow
WORKFLOW = {
    "3": {
        "inputs": {
            "prompt": "A beautiful sunset over the ocean, vibrant colors, photorealistic",
            "height": 512,
            "width": 512,
            "num_inference_steps": 4,
            "guidance_scale": 4.0,
            "seed": 42,
            "model_id": "black-forest-labs/FLUX.2-dev"
        },
        "class_type": "Flux2TPUPipeline",
    },
    "9": {
        "inputs": {
            "filename_prefix": "flux2_tpu_test",
            "images": ["3", 0]
        },
        "class_type": "SaveImage",
    }
}

def queue_prompt(workflow):
    """Submit workflow to ComfyUI queue"""
    p = {"prompt": workflow}
    response = requests.post(f"{COMFYUI_URL}/prompt", json=p)
    return response.json()

def get_history(prompt_id):
    """Get execution history"""
    response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
    return response.json()

def main():
    print("=" * 60)
    print("ComfyUI Flux.2 TPU Pipeline API Test")
    print("=" * 60)
    
    # Check if ComfyUI is running
    try:
        response = requests.get(f"{COMFYUI_URL}/system_stats")
        print(f"✓ ComfyUI is running at {COMFYUI_URL}")
    except requests.exceptions.ConnectionError:
        print(f"✗ ComfyUI is not running at {COMFYUI_URL}")
        print("  Please start ComfyUI first:")
        print("  cd /home/chrisya/ComfyUI-TPU && python3 main.py --cpu --port 8189")
        sys.exit(1)
    
    # Queue the workflow
    print("\nSubmitting workflow...")
    result = queue_prompt(WORKFLOW)
    prompt_id = result.get('prompt_id')
    
    if not prompt_id:
        print(f"✗ Failed to queue prompt: {result}")
        sys.exit(1)
    
    print(f"✓ Queued prompt: {prompt_id}")
    
    # Wait for completion
    print("\nWaiting for completion...")
    start_time = time.time()
    
    while True:
        history = get_history(prompt_id)
        if prompt_id in history:
            status = history[prompt_id].get('status', {})
            if status.get('completed', False):
                elapsed = time.time() - start_time
                print(f"✓ Completed in {elapsed:.2f}s")
                
                # Check for outputs
                outputs = history[prompt_id].get('outputs', {})
                if '9' in outputs:
                    images = outputs['9'].get('images', [])
                    if images:
                        print(f"✓ Generated {len(images)} image(s)")
                        for img in images:
                            print(f"  - {img['filename']}")
                break
            elif status.get('status_str') == 'error':
                print(f"✗ Error: {status}")
                break
        
        time.sleep(1)
        elapsed = time.time() - start_time
        print(f"  Waiting... ({elapsed:.0f}s)", end='\r')
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
