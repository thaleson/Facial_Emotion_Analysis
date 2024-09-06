import cv2
import streamlit as st
import imutils
import numpy as np
from config import EMOTIONS, FACE_MODEL_ARCHITECTURE_PATH, FACE_MODEL_WEIGHTS_PATH, HAARCASCADE_PATH
from emotion_model import EmotionModel
from face_detector import FaceDetector
from tempfile import NamedTemporaryFile
import os

def main():
    """
    Main function to run the Streamlit application for real-time facial emotion analysis.

    This function sets up the Streamlit page configuration, styles, and title. It allows the user to upload a video file,
    and upon pressing the "Start Processing" button, processes the video to detect and analyze facial emotions.
    The results are displayed in real-time, showing the detected emotions on the faces within the video.
    """
    st.set_page_config(page_title="🎬 Análise de Emoções Faciais", page_icon=":movie_camera:")

    st.markdown(
    f"""
    <style>
    {open("static/styles.css").read()}
    </style>
    """,
    unsafe_allow_html=True
)

    st.title("🎬 Análise de Emoções Faciais em Tempo Real e em Vídeos")

    st.write("""
        👋 Olá! Bem-vindo ao nosso aplicativo de detecção de emoções faciais. 
        Aqui você pode carregar um vídeo e assistir em tempo real as emoções identificadas nas faces detectadas. 😃😢😡
    """)

    video_file = st.file_uploader("📹 Selecione um arquivo de vídeo", type=["mp4", "avi", "mov"])

    if video_file:
        st.write("📂 Vídeo carregado com sucesso!")

        # Botão para iniciar o processamento do vídeo
        if st.button("▶️ Iniciar Processamento"):
            # Salvar o vídeo temporariamente
            temp_video = NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(video_file.read())
            temp_video.close()

            # Inicializar modelos
            emotion_model = EmotionModel(FACE_MODEL_ARCHITECTURE_PATH, FACE_MODEL_WEIGHTS_PATH)
            face_detector = FaceDetector(HAARCASCADE_PATH)

            # Prepare para salvar o vídeo processado
            output_file = NamedTemporaryFile(delete=False, suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file.name, fourcc, 20.0, (800, 600))  # Ajuste a resolução conforme necessário

            video_capture = cv2.VideoCapture(temp_video.name)

            while True:
                ret, frame = video_capture.read()
                if not ret:
                    st.write("📽️ Fim do vídeo.")
                    break

                frame = imutils.resize(frame, width=800)
                gray, detected_faces = face_detector.detect_faces(frame)

                for face in detected_faces:
                    (x, y, w, h) = face
                    if w > 100:
                        extracted_face = face_detector.extract_face_features(gray, face, (0.075, 0.05))
                        predictions = emotion_model.predict(extracted_face)
                        prediction_result = np.argmax(predictions)
                        expression_text = EMOTIONS[prediction_result]
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, expression_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 250, 250), 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = imutils.resize(frame, width=800)  # Ajuste a resolução conforme necessário
                out.write(frame)

            video_capture.release()
            out.release()

            st.write("✅ Processamento concluído!")
            st.video(output_file.name)  # Exibe o vídeo processado

            # Limpar arquivos temporários
            os.remove(temp_video.name)
            os.remove(output_file.name)

if __name__ == '__main__':
    main()
