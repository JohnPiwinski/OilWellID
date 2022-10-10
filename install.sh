echo "Installing Python Image Library and Pandas"
pip install -r requirements.txt
echo "Downloading Coefficients for the YOLOX classification model."
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
