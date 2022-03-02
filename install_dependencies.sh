python3.8 -m venv venv
source venv/bin/activate
apt-get install sox libsndfile1 ffmpeg
pip install -r requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip
