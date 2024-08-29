import streamlit as st
from ultralytics import YOLO
import os
import time

def yolo_object_detection_and_chat():
    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the YOLOv8 model
        model = YOLO('yolo/best (4).pt')

        # Save the uploaded image to a temporary location
        temp_image_path = "uploaded_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Perform prediction on the image
        results = model.predict(source=temp_image_path)

        # Display results
        result_images = []
        detected_classes = []
        for result in results:
            # Get the image with detections
            result_img = result.plot()  # Get the image with bounding boxes drawn
            result_images.append(result_img)

            for box in result.boxes:
                cls = box.cls.item()
                class_name = result.names[int(cls)]
                detected_classes.append(class_name)
        
        # Display the first result image with detected objects
        if result_images:
            st.image(result_images[0], caption="Detection Results", use_column_width=True)

        # Display the detected classes and automatically populate the chat input
        if detected_classes:
            st.write("Detected Objects:")
            for class_name in detected_classes:
                st.write(f"- {class_name}")
            
            # Automatically fill the text box with the first detected class
            chat_prompt = st.text_input("Ask a question about the detected object:", value=f"Tell me about {detected_classes[0]}")

            if st.button("Ask"):
                # Load other required modules for the chat
                from langchain_groq import ChatGroq
                from langchain.chains.combine_documents import create_stuff_documents_chain
                from langchain.chains import create_retrieval_chain
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_community.vectorstores import FAISS

                groq_api_key = os.getenv('GROQ_API_KEY')

                # Initialize the LLM
                llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

                # Create the prompt template
                prompt = ChatPromptTemplate.from_template(
                    """
                    Answer the questions based on the provided context only.
                    Please provide the most accurate response based on the question.
                    <context>
                    {context}
                    <context>
                    Questions: {input}
                    """
                )

                # Retrieve context for the detected class and use the chat functionality
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start = time.process_time()
                response = retrieval_chain.invoke({'input': chat_prompt})
                st.write("Response time: ", time.process_time() - start)
                st.write(response['answer'])