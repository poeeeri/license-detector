import argparse
import cv2
import logging
import torch
from pathlib import Path
from model_impl import My_LicensePlate_Model

def setup_logging():
    Path("./data").mkdir(exist_ok=True)
    logging.basicConfig(
        filename='./data/log_file.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def process_video(model, input_path, output_path, show=False):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        plates = model.detect_plates(frame)
        for p in plates:
            x1, y1, x2, y2 = p['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{p['confidence']:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        out.write(frame)
        if show:
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_count += 1
        if frame_count % 100 == 0:
            logging.info(f"Processed {frame_count} frames")
    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()
    logging.info(f"Video saved to {output_path}")

def process_camera(model, cam_id=0):
    cap = cv2.VideoCapture(cam_id)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        plates = model.detect_plates(frame)
        for p in plates:
            x1, y1, x2, y2 = p['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.imshow('License Plate Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    setup_logging()
    
    if torch.cuda.is_available():
        default_device = 'cuda'
        print("CUDA доступна")
    else:
        default_device = 'cpu'
        print("CUDA не обнаружена")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['video', 'cam'], required=True)
    parser.add_argument('--input', help='Path to video file (for video mode)')
    parser.add_argument('--output', default='output.mp4', help='Output video file')
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument('--weights', default='weights/best.pt')
    parser.add_argument('--device', default=default_device, choices=['cuda', 'cpu'], 
                        help='Device: cuda (GPU) or cpu (default: auto)')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()

    model = My_LicensePlate_Model(args.weights, device=args.device, conf_threshold=args.conf)

    if args.mode == 'video':
        if not args.input:
            logging.error("--input required for video mode")
            return
        process_video(model, args.input, args.output, show=True)
    else:
        process_camera(model, args.cam_id)

if __name__ == '__main__':
    main()