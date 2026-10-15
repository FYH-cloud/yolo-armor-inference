import cv2
from ultralytics import YOLO


def main():
    # 1. 加载模型权重
    model = YOLO('weights/best.pt')

    # 2. 读取你拍的视频
    video_path = 'origin.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ 视频打开失败，请检查 origin.mp4 是否在根目录！")
        return

    # 获取视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置输出视频
    out_path = 'result_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print("🎥 开始推理视频...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. 推理（用 CPU）
        results = model(frame, device='cpu', verbose=False)

        # 4. 画检测框和关键点
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{model.names[cls]} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                for kp in keypoints:
                    for x, y in kp:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 推理完成！结果已保存至: {out_path}")


if __name__ == '__main__':
    main()