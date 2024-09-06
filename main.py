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

        if st.button("▶️ Iniciar Processamento"):
            st.write("🔄 Iniciando processamento...")

            # Salvar o vídeo temporariamente
            temp_video = NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(video_file.read())
            temp_video.close()

            st.write(f"📂 Arquivo temporário salvo em: {temp_video.name}")

            # Inicializar modelos
            emotion_model = EmotionModel(FACE_MODEL_ARCHITECTURE_PATH, FACE_MODEL_WEIGHTS_PATH)
            face_detector = FaceDetector(HAARCASCADE_PATH)

            # Prepare para salvar o vídeo processado
            output_file = NamedTemporaryFile(delete=False, suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file.name, fourcc, 20.0, (800, 600))  # Ajuste a resolução conforme necessário

            st.write(f"📂 Arquivo de saída: {output_file.name}")

            video_capture = cv2.VideoCapture(temp_video.name)

            if not video_capture.isOpened():
                st.write("🚨 Erro ao abrir o vídeo.")
                return

            st.write("🎥 Processando vídeo...")

            frame_count = 0
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
                frame_count += 1

            video_capture.release()
            out.release()

            st.write(f"✅ Processamento concluído! {frame_count} frames processados.")

            # Testar a leitura do vídeo processado
            test_video = cv2.VideoCapture(output_file.name)
            if not test_video.isOpened():
                st.write("🚨 Erro ao abrir o vídeo processado.")
            else:
                st.write(f"📹 Exibindo o vídeo processado: {output_file.name}")
                st.video(output_file.name)  # Exibe o vídeo processado

            # Limpar arquivos temporários
            os.remove(temp_video.name)
            os.remove(output_file.name)
            
if __name__ == '__main__':
    main()
