# 刘子淳

# 生来就比别的孩子来的犟，长大就要挑比别人大的台子上。

from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import Tudui  # 假设你的模型类的定义在model.py中

app = Flask(__name__)

# 加载模型
model_path = "C:\\Users\\lenovo\\Desktop\\pytorch-tutorial-master\\pytorch-tutorial-master\\src\\tudui_3.pth"  # 指定模型文件的路径
model = Tudui()
model.load_state_dict(torch.load(model_path))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 定义苹果检测函数
def detect_apple(image_path):
    image = Image.open(image_path)
    input_image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_image)
    predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class == 0


# 定义路由
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="请选择文件")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="请选择文件")

        if file:
            file_path = "uploads/" + file.filename
            file.save(file_path)
            is_apple = detect_apple(file_path)
            result = "是" if is_apple else "不是"
            return render_template("index.html", message="预测结果：" + result)

    return render_template("index.html", message="")


if __name__ == "__main__":
    app.run(debug=True)
