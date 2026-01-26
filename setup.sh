cd ..
git clone git@github.com:565353780/camera-control.git
git clone https://github.com/NVIDIAGameWorks/kaolin.git

cd camera-control
./dev_setup.sh

pip install scipy pickle pygltflib ipyevents ipycanvas \
  rtree warp-lang

cd ../kaolin
python setup.py install
