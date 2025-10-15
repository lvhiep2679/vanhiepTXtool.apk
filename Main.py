import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
import hashlib
import numpy as np
import tensorflow.lite as tflite

class MainApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        self.input = TextInput(hint_text="Nhập mã phiên...", multiline=False, size_hint=(1,0.2))
        self.btn = Button(text="Dự đoán", size_hint=(1,0.2), background_color=(0,1,0,1))
        self.btn.bind(on_press=self.predict)
        self.result = Label(text="Kết quả sẽ hiển thị tại đây", size_hint=(1,0.6))
        
        self.layout.add_widget(self.input)
        self.layout.add_widget(self.btn)
        self.layout.add_widget(self.result)
        return self.layout

    def predict(self, instance):
        session_id = self.input.text.strip()
        if not session_id:
            self.result.text = "❌ Vui lòng nhập mã phiên!"
            return
        
        # Load TFLite model
        interpreter = tflite.Interpreter(model_path="ai_tf_model.tflite")
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Tạo features từ mã phiên
        seed_val = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        features = np.array([[seed_val % 100, (seed_val // 1000) % 100]], dtype=np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        
        prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
        rec = "🎲 Tài" if prob >= 0.5 else "🎲 Xỉu"
        
        self.result.text = f"🔑 Mã phiên: {session_id}\n🤖 AI dự đoán: {rec}"

if __name__ == "__main__":
    MainApp().run()
