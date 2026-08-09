from ultralytics import YOLO

# 1. நீங்க Train செய்த Model-ஐ load பண்றது
model = YOLO('yolov8x.pt')

# 2. Image-ஐ இன்புட்டாக கொடுத்து, ஸ்க்ரீனில் பார்க்கச் சொல்வது (show=True)
# (உங்கள் படம் ஃபைல் பெயர் 'test_image.jpg' இல்லை என்றால் மாற்றி எழுதிக்கொள்ளுங்கள்)
results = model.predict(source='test_image.jpg', show=True, save=True, conf=0.5)

print("Image Testing Completed! 'runs' folder-ஐ செக் பண்ணுங்க.")