import cv2
import os
from tqdm import tqdm

def make_frames(input_folder, output_folder, step=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    extracted_count = 0

    # Проходим по всем подкаталогам и файлам в input_folder
    for root, dirs, files in os.walk(input_folder):
        # Отбираем все файлы в текущей директории
        video_files = [f for f in files]
        for file in video_files:
            video_path = os.path.join(root, file)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if fps == 0 or fps is None:
                print(f"Не удалось получить FPS для {video_path}. Пропуск файла.")
                continue

            frame_interval = int(fps * step)
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                print(f"Не удалось получить общее количество кадров для {video_path}.")
                total_frames = None

            print(f"Обработка видео: {video_path}")
            with tqdm(total=total_frames, unit='кадр', desc=file, leave=False) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % frame_interval == 0:
                        # Создаем уникальное имя файла для каждого кадра
                        relative_path = os.path.relpath(root, input_folder)
                        date_folder = os.path.basename(relative_path)
                        frame_filename = os.path.join(
                            output_folder,
                            f"{date_folder}_{os.path.splitext(file)[0]}_frame_{frame_count}.jpg"
                        )
                        cv2.imwrite(frame_filename, frame)
                        extracted_count += 1
                    frame_count += 1
                    pbar.update(1)
            cap.release()
            print(f"{video_path} : Извлечено {extracted_count} кадров.\n")

if __name__ == '__main__':
    input_folder = '../ITMO-HSE-MLBD-LW-3'
    output_folder = './frames'
    make_frames(input_folder, output_folder, step=0.5)